"""CPU-only integration contracts for the optional packaged Metal backend."""
from pathlib import Path
import sys
from types import ModuleType
import xml.etree.ElementTree as ET

import pytest
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")


def test_resolver_uses_package_without_sys_path_or_generic_src(monkeypatch):
    from mjlab_microduck.native_gpu.backend_selection import resolve_metal_backend
    monkeypatch.delenv('MICRODUCK_METAL_BACKEND_ROOT', raising=False)
    unrelated = ModuleType('src')
    monkeypatch.setitem(sys.modules, 'src', unrelated)
    before = list(sys.path)
    backend, paths = resolve_metal_backend()
    assert backend.__module__ == 'mjlab_microduck.native_gpu.metal.metal_simulation_adapter'
    assert sys.path == before
    assert sys.modules['src'] is unrelated
    assert all(Path(p).is_file() for p in paths.values())
    assert Path(paths['shader']).name == 'physics_slice.metal'


def test_old_experiment_override_fails_clearly(monkeypatch, tmp_path):
    from mjlab_microduck.native_gpu.backend_selection import resolve_metal_backend
    monkeypatch.setenv('MICRODUCK_METAL_BACKEND_ROOT', str(tmp_path))
    with pytest.raises(RuntimeError, match='Unset MICRODUCK_METAL_BACKEND_ROOT'):
        resolve_metal_backend()


def test_packaged_model_and_all_mesh_assets_load_on_cpu():
    from mjlab_microduck.native_gpu.metal.canonical_model_loader import CANONICAL_XML_PATH, load_canonical_model
    xml = ET.parse(CANONICAL_XML_PATH).getroot()
    mesh_dir = (CANONICAL_XML_PATH.parent / xml.find('compiler').get('meshdir')).resolve()
    for mesh in xml.findall('./asset/mesh'):
        assert (mesh_dir / mesh.get('file')).is_file()
    model = load_canonical_model().model
    assert (model.nbody, model.ngeom, model.nq, model.nv, model.nu) == (17, 76, 21, 20, 14)


def test_shader_resolution_does_not_compile_or_allocate_gpu(monkeypatch):
    import torch
    from mjlab_microduck.native_gpu.backend_selection import resolve_metal_backend
    monkeypatch.delenv('MICRODUCK_METAL_BACKEND_ROOT', raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU preflight must not compile GPU shaders')
    monkeypatch.setattr(torch.mps, 'compile_shader', forbidden)
    _, paths = resolve_metal_backend()
    assert 'kernel void' in Path(paths['shader']).read_text()
