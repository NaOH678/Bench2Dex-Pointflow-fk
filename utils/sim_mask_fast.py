"""Allocation-free per-triangle CPU rasterization, preserving reference mask semantics."""
import numpy as np
from numba import njit


@njit(cache=True)
def rasterize_scalar(triangles_camera, triangle_labels, K, height, width):
    depth=np.full((height,width),np.inf,np.float32)
    labels=np.zeros((height,width),np.int32)
    for i in range(len(triangles_camera)):
        z0=triangles_camera[i,0,2];z1=triangles_camera[i,1,2];z2=triangles_camera[i,2,2]
        if min(z0,z1,z2)<=1e-5:continue
        x0=triangles_camera[i,0,0]/z0*K[0,0]+K[0,2]
        x1=triangles_camera[i,1,0]/z1*K[0,0]+K[0,2]
        x2=triangles_camera[i,2,0]/z2*K[0,0]+K[0,2]
        y0=triangles_camera[i,0,1]/z0*K[1,1]+K[1,2]
        y1=triangles_camera[i,1,1]/z1*K[1,1]+K[1,2]
        y2=triangles_camera[i,2,1]/z2*K[1,1]+K[1,2]
        xmin=max(0,int(np.ceil(min(x0,x1,x2))));xmax=min(width-1,int(np.floor(max(x0,x1,x2))))
        ymin=max(0,int(np.ceil(min(y0,y1,y2))));ymax=min(height-1,int(np.floor(max(y0,y1,y2))))
        denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
        if abs(denom)<1e-12:continue
        for v in range(ymin,ymax+1):
            for u in range(xmin,xmax+1):
                a=((y1-y2)*(u-x2)+(x2-x1)*(v-y2))/denom
                b=((y2-y0)*(u-x2)+(x0-x2)*(v-y2))/denom
                c=1-a-b
                if min(a,b,c)<-1e-6:continue
                z=1/(a/z0+b/z1+c/z2)
                if z<depth[v,u]:
                    depth[v,u]=z;labels[v,u]=triangle_labels[i]
    return depth,labels
