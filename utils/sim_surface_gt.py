"""Independent offline surface labels: URDF FK and USD visual geometry.

Never import this module from observation extraction or model input selection.
Transforms use column-vector conventions, meters and xyzw quaternions.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(R=None, t=None):
    out = np.eye(4)
    if R is not None:
        out[:3, :3] = R
    if t is not None:
        out[:3, 3] = t
    return out


def apply(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def pose_matrix(pose):
    return transform(Rotation.from_quat(pose[3:]).as_matrix(), pose[:3])


class KinematicTree:
    def __init__(self, joints, root='base_link'):
        self.joints, self.root = joints, root

    @classmethod
    def from_urdf(cls, path):
        joints = {}
        for j in ET.parse(path).getroot().findall('joint'):
            origin, axis = j.find('origin'), j.find('axis')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            joints[j.get('name')] = dict(parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                before=transform(Rotation.from_euler('xyz', rpy).as_matrix(), xyz), after=np.eye(4),
                axis=np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.array([1., 0, 0]),
                type=j.get('type'))
        roots = {j["parent"] for j in joints.values()} - {j["child"] for j in joints.values()}
        if len(roots) != 1:
            raise ValueError(f"Expected one kinematic root, got {roots}")
        return cls(joints, root=next(iter(roots)))

    @classmethod
    def from_usd(cls, stage):
        from pxr import UsdPhysics
        def local(pos, q):
            return transform(Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix(), pos)
        joints = {}
        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            parents, children = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if not parents or not children:
                continue
            axis = p.GetAttribute('physics:axis').Get()
            kind = {'PhysicsRevoluteJoint': 'revolute', 'PhysicsFixedJoint': 'fixed',
                    'PhysicsPrismaticJoint': 'prismatic'}.get(p.GetTypeName())
            if kind is None:
                raise ValueError(f'Unsupported joint {p.GetPath()}')
            joints[p.GetName()] = dict(parent=parents[0].name, child=children[0].name,
                before=local(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                after=np.linalg.inv(local(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())),
                axis=np.eye(3)['XYZ'.index(axis)] if axis else np.array([1., 0, 0]), type=kind)
        roots = {j["parent"] for j in joints.values()} - {j["child"] for j in joints.values()}
        if len(roots) != 1:
            raise ValueError(f"Expected one kinematic root, got {roots}")
        return cls(joints, root=next(iter(roots)))

    def fk(self, values):
        nodes, pending = {self.root: np.eye(4)}, dict(self.joints)
        while pending:
            progress = False
            for name, j in list(pending.items()):
                if j['parent'] not in nodes:
                    continue
                kind = j['type']
                if kind != 'fixed' and name not in values:
                    raise ValueError(f'Missing recorded joint: {name}')
                motion = np.eye(4)
                if kind in ('revolute', 'continuous'):
                    motion[:3, :3] = Rotation.from_rotvec(j['axis']*values[name]).as_matrix()
                elif kind == 'prismatic':
                    motion[:3, 3] = j['axis']*values[name]
                elif kind != 'fixed':
                    raise ValueError(kind)
                nodes[j['child']] = nodes[j['parent']] @ j['before'] @ motion @ j['after']
                del pending[name]
                progress = True
            if not progress:
                raise ValueError(f'Disconnected joints: {list(pending)}')
        return nodes


def mesh_triangles(prim):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces, start = [], 0
    for count in counts:
        face = indices[start:start+count]
        faces.extend((face[0], face[k], face[k+1]) for k in range(1, count-1))
        start += count
    return points[np.asarray(faces, int)]


def robot_visuals(stage):
    from pxr import Usd, UsdGeom
    root = stage.GetDefaultPrim().GetPath()
    meshes = []
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or '/visuals/' not in str(p.GetPath()):
            continue
        link = str(p.GetPath()).removeprefix(str(root)+'/').split('/')[0]
        Tmesh = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        Tlink = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(root.AppendChild(link))).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        tri = apply(np.linalg.inv(Tlink) @ Tmesh, mesh_triangles(p))
        meshes.append(dict(name=link, path=str(p.GetPath()), triangles=tri,
                           group='hand' if 'finger' in link or 'palm' in link else 'arm'))
    return meshes


def object_visuals(stage, scale, name):
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetDefaultPrim()
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid) != 1 or rigid[0] != root:
        raise ValueError(f'{name}: expected one rigid body at asset root, got {[str(p.GetPath()) for p in rigid]}')
    root_T = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    meshes = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(p)
        if imageable.ComputeVisibility() == 'invisible' or imageable.ComputePurpose() in ('guide', 'proxy'):
            continue
        # Collision-only meshes are hidden in the shipped assets.
        T = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        tri = apply(np.linalg.inv(root_T) @ T, mesh_triangles(p))*np.asarray(scale)
        meshes.append(dict(name=name, path=str(p.GetPath()), triangles=tri, group='object'))
    return meshes


def first_ray_hits(triangles, origin, directions, batch=128):
    import trimesh
    mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(triangles.size//3).reshape(-1, 3), process=False)
    hit = np.full((len(directions), 3), np.nan)
    face = np.full(len(directions), -1, int)
    for begin in range(0, len(directions), batch):
        dirs = directions[begin:begin+batch]
        loc, ray_id, triangle_id = mesh.ray.intersects_location(np.broadcast_to(origin, dirs.shape), dirs, multiple_hits=False)
        hit[begin+ray_id], face[begin+ray_id] = loc, triangle_id
    return hit, face
