"""Execute a JAX graph in GPU chunks separated by control-flow boundaries.

Numerical equations are unchanged. Python schedules loops/branches by reading
scalar conditions; all array operations remain on the selected JAX device.
This avoids the MPS backend's monolithic nested-loop compilation path.
"""

import jax
import jax.numpy as jnp
from jax.extend import core


def _var(value):
    return isinstance(value, core.Var)


def _graph(value):
    return value.jaxpr if isinstance(value, core.ClosedJaxpr) else value


def _nested(equation):
    for value in equation.params.values():
        for item in value if isinstance(value, (tuple, list)) else [value]:
            if isinstance(item, (core.Jaxpr, core.ClosedJaxpr)):
                yield _graph(item)


def _has_control(graph):
    return any(
        e.primitive.name in ("while", "scan", "cond")
        or any(_has_control(g) for g in _nested(e))
        for e in graph.eqns
    )


class Graph:
    def __init__(self, closed):
        self.graph = closed.jaxpr
        self.constants = closed.consts
        graph = self.graph
        groups, pending = [], []
        for equation in graph.eqns:
            boundary = equation.primitive.name in ("while", "scan", "cond") or any(
                _has_control(g) for g in _nested(equation)
            )
            if boundary:
                if pending:
                    groups.append(pending)
                    pending = []
                groups.append(equation)
            else:
                pending.append(equation)
        if pending:
            groups.append(pending)
        live = {v for v in graph.outvars if _var(v)}
        instructions = []
        for group in reversed(groups):
            if not isinstance(group, list):
                inputs, outputs = group.invars, group.outvars
                run = self._control(group)
            else:
                written = {v for e in group for v in e.outvars if _var(v)}
                inputs, seen = [], set()
                for e in group:
                    for v in e.invars:
                        if _var(v) and v not in written and v not in seen:
                            inputs.append(v)
                            seen.add(v)
                outputs = [v for e in group for v in e.outvars if v in live]
                chunk = graph.replace(
                    constvars=[], invars=inputs, outvars=outputs, eqns=group
                )
                run = jax.jit(core.jaxpr_as_fun(core.ClosedJaxpr(chunk, [])))
            keep = live.copy()
            live.difference_update(v for v in outputs if _var(v))
            live.update(v for v in inputs if _var(v))
            instructions.append((inputs, outputs, run, keep))
        self.instructions = list(reversed(instructions))

    def _control(self, equation):
        name, params = equation.primitive.name, equation.params
        if name == "while":
            cond = Graph(core.ClosedJaxpr(params["cond_jaxpr"], []))
            body = Graph(core.ClosedJaxpr(params["body_jaxpr"], []))
            nc, nb = params["cond_nconsts"], params["body_nconsts"]

            def run(*args):
                c, b, carry = args[:nc], args[nc : nc + nb], args[nc + nb :]
                while bool(cond(*c, *carry)[0]):
                    carry = body(*b, *carry)
                return carry

            return run
        if name == "scan":
            body = Graph(params["jaxpr"])
            length = params["length"]

            def run(*args):
                constants, carry, xs = map(list, params["ft_in"].update(args).unpack())
                accumulated = None
                indices = (
                    range(length - 1, -1, -1) if params["reverse"] else range(length)
                )
                for index in indices:
                    result = body(*constants, *carry, *(x[index] for x in xs))
                    carry, ys = map(list, params["ft_out"].update(result).unpack())
                    if accumulated is None:
                        accumulated = [[] for _ in ys]
                    for values, y in zip(accumulated, ys):
                        values.append(y)
                if length == 0:
                    raise NotImplementedError(
                        "Zero-length staged scan is not qualified"
                    )
                ys = [
                    jnp.stack(list(reversed(v)) if params["reverse"] else v)
                    for v in accumulated
                ]
                return [*carry, *ys]

            return run
        if name == "cond":
            branches = [Graph(g) for g in params["branches"]]
            return lambda index, *args: branches[int(index)](*args)
        if name == "jit":
            return Graph(params["jaxpr"])
        raise NotImplementedError(f"Unsupported nested control primitive {name}")

    def __call__(self, *args):
        graph = self.graph
        env = dict(zip(graph.constvars, self.constants))
        env.update(zip(graph.invars, args))

        def read(v):
            return env[v] if _var(v) else v.val

        for inputs, outputs, run, keep in self.instructions:
            values = run(*(read(v) for v in inputs))
            env.update((v, value) for v, value in zip(outputs, values) if _var(v))
            env = {v: value for v, value in env.items() if v in keep}
        return [read(v) for v in graph.outvars]


class StagedJit:
    def __init__(self, function):
        self.function = function
        self.graph = None
        self.input_tree = None
        self.output_tree = None

    def __call__(self, *args):
        if self.graph is None:
            graph, shape = jax.make_jaxpr(self.function, return_shape=True)(*args)
            self.graph = Graph(graph)
            self.input_tree = jax.tree.structure(args)
            self.output_tree = jax.tree.structure(shape)
        flat, tree = jax.tree.flatten(args)
        if tree != self.input_tree:
            raise ValueError("Staged function input tree changed")
        return jax.tree.unflatten(self.output_tree, self.graph(*flat))
