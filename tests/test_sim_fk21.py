from pathlib import Path
import numpy as np
from utils.sim_surface_gt import KinematicTree
from utils.sim_fk21 import positions_from_transforms


def test_distal_joint_moves_tip_but_not_its_own_origin_or_other_hand():
    urdf = Path(__file__).resolve().parents[2]/'dex2bench_dataset/Robots_p/ur5+wuji/urdf/Multi_UR5_wuji_with_flange.urdf'
    tree = KinematicTree.from_urdf(urdf)
    q = {n:0. for n,j in tree.joints.items() if j['type'] != 'fixed'}
    before = positions_from_transforms(tree.fk(q))
    q['right_finger2_joint4'] = .2
    after = positions_from_transforms(tree.fk(q))
    assert np.linalg.norm(after[1,8]-before[1,8]) > .001
    np.testing.assert_allclose(after[1,7],before[1,7],atol=1e-12)
    np.testing.assert_array_equal(after[0],before[0])
