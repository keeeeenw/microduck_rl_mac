from types import SimpleNamespace
import mujoco
import numpy as np
import torch

import pytest
pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from mjlab_microduck.native_gpu.metal.metal_simulation_adapter import UnifiedMetalSimulation


XML = '''<mujoco><worldbody>
  <body pos="0 0 0.3"><freejoint/><geom name="a" type="sphere" size="0.1" mass="1"/></body>
  <body pos="0.15 0 0.3"><freejoint/><geom name="b" type="sphere" size="0.1" mass="1"/></body>
  <geom name="unrelated" pos="3 0 0" type="sphere" size="0.1"/>
</worldbody></mujoco>'''


def make_sim(hit):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    B = 2
    sim = object.__new__(UnifiedMetalSimulation)
    sim.mj_model = m
    sim.num_envs = B
    sim.device = 'cpu'
    sim._fk_cache_signature = None
    sim.slice = SimpleNamespace(body_xmat=torch.tensor(np.repeat(d.xmat[None], B, axis=0).copy(), dtype=torch.float32), body_xpos=torch.tensor(np.repeat(d.xpos[None], B, axis=0).copy(), dtype=torch.float32))
    sim.data = SimpleNamespace(qpos=torch.tensor(np.repeat(m.qpos0[None], B, axis=0).copy(), dtype=torch.float32), geom_xpos=torch.zeros(B,m.ngeom,3), geom_xmat=torch.zeros(B,m.ngeom,9))
    sim.slice.compute_forward_kinematics = lambda qpos: None
    sim.geom_bodyid = torch.tensor(m.geom_bodyid, dtype=torch.long)
    sim.geom_pos = torch.tensor(m.geom_pos, dtype=torch.float32)
    sim.geom_mat = torch.eye(3).repeat(m.ngeom,1,1)
    first=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'a')
    second=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'b')
    for name in ('left_leg','right_leg','left_foot','right_foot','trunk'):
        setattr(sim, f'{name}_geom_id', first if name.startswith('left') or name == 'trunk' else second)
        setattr(sim, f'center_{name}', torch.zeros(3))
        setattr(sim, f'e_{name}', torch.ones(3))
    sim._sat_obb_intersection = lambda *args: torch.as_tensor([hit] * B if isinstance(hit, bool) else hit, dtype=torch.bool)
    pair=tuple(sorted((first,second)))
    sim._sc_geom_pairs=frozenset([pair])
    sim._sc_friction_geom_ids=np.array(pair, dtype=np.int64)
    sim._sc_friction_geom_slots={pair[0]:0,pair[1]:1}
    sim._sc_cpu_model=m
    sim._sc_cpu_data=mujoco.MjData(m)
    friction=torch.tensor([[[9,9,9],[.2,.01,.01],[.8,.02,.02]], [[9,9,9],[.9,.03,.03],[.1,.04,.04]]])
    sim.model=SimpleNamespace(geom_friction=friction)
    sim.max_extra_contacts=4
    sim._current_self_col_count=torch.zeros(B)
    sim._extra_ncon=torch.zeros(B,dtype=torch.int32)
    sim._extra_c_pos=torch.zeros(B,4,3)
    sim._extra_c_dist=torch.zeros(B,4)
    sim._extra_c_b1=torch.zeros(B,4,dtype=torch.int32)
    sim._extra_c_b2=torch.zeros(B,4,dtype=torch.int32)
    sim._extra_c_frame=torch.zeros(B,4,9)
    sim._extra_c_fric=torch.zeros(B,4,2)
    sim.collision_qpos_staging_bytes=0
    sim.collision_fric_staging_bytes=0
    sim.collision_evaluations_count=0
    sim.extra_contact_overflow_count=0
    sim.collision_transfer=SimpleNamespace(bytes=0,seconds=0.0)
    sim.collision_narrowphase_seconds=0.0
    sim.collision_fallback_branch_seconds=0.0
    return sim


def test_accepted_mujoco_contacts_match_original_full_friction_payload():
    sim=make_sim(True)
    payload=sim._detect_self_contacts()
    assert payload is not None
    assert sim._extra_ncon.tolist() == [1,1]
    assert sim.collision_fric_staging_bytes == 2 * 2 * 3 * 4
    m=sim.mj_model
    for env_id in range(2):
        d=mujoco.MjData(m)
        d.qpos[:]=sim.data.qpos[env_id].numpy()
        mujoco.mj_kinematics(m,d)
        mujoco.mj_collision(m,d)
        accepted=[d.contact[i] for i in range(d.ncon) if tuple(sorted((int(d.contact[i].geom1),int(d.contact[i].geom2)))) in sim._sc_geom_pairs and d.contact[i].dist <= 0]
        assert len(accepted)==1
        c=accepted[0]
        np.testing.assert_allclose(payload['pos'][env_id,0].numpy(),c.pos,atol=1e-6)
        np.testing.assert_allclose(payload['dist'][env_id,0].numpy(),c.dist,atol=1e-6)
        np.testing.assert_allclose(payload['frame'][env_id,0].numpy(),c.frame,atol=1e-6)
        assert payload['body1'][env_id,0] == m.geom_bodyid[c.geom1]
        assert payload['body2'][env_id,0] == m.geom_bodyid[c.geom2]
        full=sim.model.geom_friction[env_id].numpy()
        expected=max(float(full[c.geom1,0]),float(full[c.geom2,0]))
        np.testing.assert_allclose(payload['friction'][env_id,0].numpy(),[expected,expected])


def test_no_candidate_skips_narrowphase_and_friction_staging():
    sim=make_sim(False)
    assert sim._detect_self_contacts() is None
    assert sim.collision_fric_staging_bytes == 0
    assert sim.collision_evaluations_count == 0


def test_mixed_candidate_rows_scatter_and_clear_stale_payload():
    sim=make_sim([True,False])
    first=sim._detect_self_contacts()
    assert first is not None
    assert sim._extra_ncon.tolist() == [1,0]
    assert sim._extra_c_fric[1].count_nonzero() == 0
    sim._sat_obb_intersection=lambda *args: torch.tensor([False,True])
    second=sim._detect_self_contacts()
    assert second is not None
    assert sim._extra_ncon.tolist() == [0,1]
    assert sim._extra_c_fric[0].count_nonzero() == 0
    assert sim._extra_c_fric[1,0,0] == .9
    sim._sat_obb_intersection=lambda *args: torch.tensor([False,False])
    assert sim._detect_self_contacts() is None
    assert sim._extra_ncon.tolist() == [0,0]
    assert sim._current_self_col_count.tolist() == [0,0]
