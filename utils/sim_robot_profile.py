"""Robot-specific offline geometry profiles, resolved from repository spawners."""
from pathlib import Path
import ast, json
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from utils.sim_surface_gt import KinematicTree, transform
ROOT=Path(__file__).resolve().parents[1]

def constants(path):
    result={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):result[target.id]=value
    return result

def profile(source,assets):
    key=source['meta/robot_key'][()]
    if isinstance(key,bytes):key=key.decode()
    supported={'multi_ur5_wuji_with_flange','multi_ur5_rh56dfx_with_flange','multi_ur5_rh5dg2_with_flange','multi_jaka_zu7_dexhand021_with_flange'}
    if key not in supported:raise ValueError('Unsupported offline robot '+key)
    row=yaml.safe_load((ROOT/'robots/active_dof_maps.yml').read_text())['robots'][key]
    urdf=Path(assets)/('Robots_p/'+row['urdf_path'].split('Robots_p/',1)[1]);usd=Path(assets)/('Robots_p/'+row['usd_path'].split('Robots_p/',1)[1])
    c=constants(ROOT/'robots'/f'{key}.py');table=constants(ROOT/'build/table_geometry.py')
    base=transform(Rotation.from_euler('xyz',c['ROBOT_RPY_DEG'],degrees=True).as_matrix(),[c.get('BASE_X_OFFSET',.5),table['ROBOT_BACK_REFERENCE_Y']+c['BACK_EDGE_MARGIN'],table['ROBOT_SUPPORT_TABLE_HEIGHT']+c['BASE_Z_OFFSET']])
    tree=KinematicTree.from_urdf(urdf)
    mapping=[];roots=[]
    for side,short in [('left','l'),('right','r')]:
        def point(link,other=None):return dict(link=link,**({'midpoint_to':other} if other else {}))
        if 'wuji' in key:
            root=f'{side}_palm_link';points=[point(root)]
            for i in range(1,6):points += [point(f'{side}_finger{i}_link{j}') for j in (2,3,4)]+[point(f'{side}_finger{i}_tip_link')]
        elif 'rh56dfx' in key:
            root=f'{short}_base_link';points=[point(root)]+[point(f'{side}_thumb_{j}') for j in (2,3,4)]+[point(f'{side}_thumb_tip')]
            for finger in ['index','middle','ring','little']:
                a=f'{side}_{finger}_2';tip=f'{side}_{finger}_tip'
                points += [point(f'{side}_{finger}_1'),point(a),point(a,tip),point(tip)]
        elif 'rh5dg2' in key:
            root=f'{side}_hand_base';points=[point(root)]
            for finger in ['thumb','index','middle','ring','pinky']:
                points += [point(f'{side}_{finger}_{j}') for j in ['mcp','pip','dip','force_sensor']]
        else:
            root=f'{short}_p_link0';points=[point(root)]
            for i in range(1,6):points += [point(f'{short}_f_link{i}_{j}') for j in (2,3,4)]+[point(f'{short}_f_link{i}_tip')]
        assert len(points)==21;mapping.append(points);roots.append(root)
    hands=set(roots)
    while True:
        new=hands|{j['child'] for j in tree.joints.values() if j['parent'] in hands}
        if new==hands:break
        hands=new
    return dict(robot_key=key,urdf=urdf,usd=usd,base=base,tree=tree,hand_links=hands,mapping=mapping)

def positions(transforms,mapping):
    def at(p):
        a=transforms[p['link']][:3,3]
        return (a+transforms[p['midpoint_to']][:3,3])/2 if 'midpoint_to' in p else a
    return np.array([[at(p) for p in hand] for hand in mapping])

def document(p):
    return dict(schema='bench2dex_robot_fk21_v1',robot_key=p['robot_key'],urdf=str(p['urdf'].resolve()),usd=str(p['usd'].resolve()),world_from_robot_base=p['base'].tolist(),mapping=p['mapping'],hand_links=sorted(p['hand_links']),units='metre',convention='robot landmark proxies; RH56DFX non-thumb DIP is midpoint of distal hinge and tip; RH5DG2 tip uses force-sensor origin; Wuji mapping unchanged')
