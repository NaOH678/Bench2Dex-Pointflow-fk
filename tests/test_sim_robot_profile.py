from pathlib import Path
import numpy as np
from utils.sim_surface_gt import KinematicTree
from utils.sim_robot_profile import positions


def test_fk_detects_non_wuji_root(tmp_path):
    p=tmp_path/'robot.urdf'
    p.write_text('<robot name="test"><link name="Base"/><link name="tip"/><joint name="j" type="prismatic"><parent link="Base"/><child link="tip"/><origin xyz="1 0 0"/><axis xyz="0 1 0"/></joint></robot>')
    tree=KinematicTree.from_urdf(p)
    assert tree.root=='Base'
    np.testing.assert_allclose(tree.fk({'j':2})['tip'][:3,3],[1,2,0])


def test_rh56_proxy_midpoint_uses_two_current_link_positions():
    a=np.eye(4);b=np.eye(4);b[:3,3]=[0,0,2]
    mapping=[[{'link':'a'},{'link':'a','midpoint_to':'b'},{'link':'b'}]]
    np.testing.assert_allclose(positions({'a':a,'b':b},mapping)[0],[[0,0,0],[0,0,1],[0,0,2]])
