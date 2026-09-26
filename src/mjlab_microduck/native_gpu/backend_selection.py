"""Resolve the bundled experimental Metal backend without modifying sys.path."""

import importlib
import os
from pathlib import Path


def resolve_metal_backend():
    # Old experiment overrides must not silently select a different implementation.
    if os.environ.get("MICRODUCK_METAL_BACKEND_ROOT"):
        raise RuntimeError(
            "Metal is now bundled with mjlab_microduck. Unset "
            "MICRODUCK_METAL_BACKEND_ROOT and use --physics metal."
        )
    package = __package__ + ".metal"
    modules = {
        "adapter": importlib.import_module(package + ".metal_simulation_adapter"),
        "slice": importlib.import_module(package + ".representative_physics_slice"),
        "kernel_manager": importlib.import_module(package + ".metal_kernel_manager"),
        "model_loader": importlib.import_module(package + ".canonical_model_loader"),
    }
    root = Path(__file__).resolve().parent
    paths = {key: str(Path(module.__file__).resolve()) for key, module in modules.items()}
    paths["rl"] = str(root / "environment.py")
    paths["shader"] = str(modules["slice"].SHADER_PATH.resolve())
    paths["canonical_xml"] = str(modules["model_loader"].CANONICAL_XML_PATH.resolve())
    for key, value in paths.items():
        path = Path(value)
        if not path.is_file() or not path.is_relative_to(root):
            raise RuntimeError(f"Unexpected or missing packaged Metal {key}: {path}")
    return modules["adapter"].UnifiedMetalSimulation, paths
