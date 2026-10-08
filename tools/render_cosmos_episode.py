"""Export recorded closed-loop RGB to MP4 at an explicit recording frame rate."""
import argparse
from pathlib import Path
import subprocess
import cv2
import h5py
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--episode',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--fps',type=float,default=20)
p.add_argument('--three-views',action='store_true')
p.add_argument('--threads',type=int,default=2)
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.episode) as f:
    cameras=['cam_overhead','cam_wrist_left','cam_wrist_right'] if a.three_views else ['cam_overhead']
    datasets=[f[f'cameras/{c}/rgb'] for c in cameras]
    n=len(datasets[0])
    if not n or any(len(d)!=n for d in datasets): raise ValueError('Missing or unaligned camera frames')
    def frame(i):
        ims=[cv2.imdecode(d[i],cv2.IMREAD_COLOR) for d in datasets]
        if any(im is None for im in ims): raise ValueError(f'Missing RGB frame {i}')
        if not a.three_views: return ims[0]
        top=ims[0];h,w=top.shape[:2]
        bottom=np.concatenate([cv2.resize(im,(w//2,h//2),interpolation=cv2.INTER_AREA) for im in ims[1:]],axis=1)
        return np.concatenate([top,bottom],axis=0)
    first=frame(0);h,w=first.shape[:2]
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}','-framerate',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-threads',str(a.threads),'-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    try:
        for i in range(n): proc.stdin.write(frame(i).tobytes())
    finally:
        proc.stdin.close()
    if proc.wait(): raise RuntimeError('ffmpeg failed')
    print(f'{a.output}: {n} frames, {n/a.fps:.2f}s at {a.fps}fps')
