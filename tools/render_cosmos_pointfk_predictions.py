"""Offline predicted Point/FK overlay on recorded head RGB (no model or simulator)."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np


EDGES = [(0, k) for k in (1, 5, 9, 13, 17)]
EDGES += [(k + j, k + j + 1) for k in (1, 5, 9, 13, 17) for j in range(3)]


def positions(archive, name, offset):
    anchor = archive[name + '_anchor_xyz']
    if not 0 <= offset <= 32:
        raise ValueError('Prediction offset outside the 32-step horizon')
    return anchor if offset == 0 else anchor + archive[name + '_displacement'][offset - 1]


def project(xyz, intrinsic, width, height):
    projected = np.asarray(xyz) @ intrinsic.T
    valid = np.isfinite(projected).all(-1) & (projected[:, 2] > 1e-6)
    uv = np.zeros((len(xyz), 2), np.float64)
    uv[valid] = projected[valid, :2] / projected[valid, 2:]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return np.rint(uv).astype(np.int64), valid


def query_for_frame(starts, frame, adoption_steps=None):
    if adoption_steps is None:
        index = int(np.searchsorted(starts, frame, side='right') - 1)
    else:
        eligible = [i for i, step in enumerate(adoption_steps) if step is not None and step <= frame]
        index = eligible[-1] if eligible else -1
    if index < 0 or not 0 <= frame - starts[index] <= 32:
        raise ValueError(f'No prediction covers recorded frame {frame}')
    return index, int(frame - starts[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True, help='One episode directory of query_*.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-index', type=int, help='Render one complete 32-step forecast instead of latest-query overlay')
    parser.add_argument('--timing', type=Path, help='Select the actually adopted async prediction at each control frame')
    parser.add_argument('--record-stride', type=int, default=1)
    args = parser.parse_args()
    if args.record_stride < 1:
        parser.error('record-stride must be positive')
    files = sorted(args.predictions.glob('query_*.npz'))
    if not files:
        parser.error('No saved predictions')
    archives, metadata = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            archives.append({k: data[k] for k in data.files})
        metadata.append(json.loads(str(archives[-1]['metadata_json'])))
    if any(m['schema'] != 'cosmos_live_pointfk_prediction_v1' or m['fps'] != 20 for m in metadata):
        raise ValueError('Unsupported prediction contract')
    starts = np.array([round(m['sim_time_sec'] * 20) for m in metadata])
    if np.any(np.diff(starts) <= 0):
        raise ValueError('Query times must be strictly increasing')
    adoption_steps = None
    if args.timing is not None:
        timing = json.loads(args.timing.read_text())
        queries = timing['queries']
        if len(queries) != len(archives) or any(q['observation_step'] != s for q, s in zip(queries, starts)):
            raise ValueError('Prediction and timing query indices do not match')
        adoption_steps = [q.get('adopt_step') for q in queries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with h5py.File(args.episode) as source:
        rgb = source['cameras/cam_overhead/rgb']
        frame_ids = np.arange(len(rgb)) * args.record_stride
        if args.query_index is not None:
            if not 0 <= args.query_index < len(archives):
                raise ValueError('query-index out of range')
            start = starts[args.query_index]
            rows = np.flatnonzero((frame_ids >= start) & (frame_ids <= start + 32))
        else:
            rows = np.arange(len(rgb))
        width, height = map(int, archives[0]['image_size_wh'])
        command = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
                   '-video_size', f'{2 * width}x{height + 64}', '-framerate', str(20 / args.record_stride),
                   '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for row in rows:
                frame_id = int(frame_ids[row])
                if args.query_index is None:
                    index, offset = query_for_frame(starts, frame_id, adoption_steps)
                else:
                    index, offset = args.query_index, int(frame_id - starts[args.query_index])
                archive = archives[index]
                frame = cv2.imdecode(rgb[row], cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError(f'Invalid head RGB frame {row}')
                overlay = frame.copy()
                uv, valid = project(positions(archive, 'pointflow', offset), archive['intrinsic'], width, height)
                # Spatial color remains constant within each query, as do point identities.
                anchor_uv = archive['pointflow_anchor_uv']
                hsv = np.zeros((len(uv), 1, 3), np.uint8)
                hsv[:, 0, 0] = np.clip(anchor_uv[:, 0] / width * 179, 0, 179).astype(np.uint8)
                hsv[:, 0, 1:] = 255
                colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
                for j in np.flatnonzero(valid):
                    cv2.circle(overlay, tuple(uv[j]), 1, tuple(map(int, colors[j])), -1, cv2.LINE_AA)
                uv, valid = project(positions(archive, 'fk', offset), archive['intrinsic'], width, height)
                for hand, color in enumerate([(40, 220, 255), (255, 140, 30)]):
                    for a, b in EDGES:
                        a += hand * 21; b += hand * 21
                        if valid[a] and valid[b]:
                            cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)
                    for j in range(hand * 21, (hand + 1) * 21):
                        if valid[j]:
                            cv2.circle(overlay, tuple(uv[j]), 2, color, -1, cv2.LINE_AA)
                canvas = np.zeros((height + 64, 2 * width, 3), np.uint8)
                canvas[64:, :width] = frame
                canvas[64:, width:] = overlay
                lines = [f'Recorded RGB | frame {frame_id} | {frame_id / 20:.2f}s',
                         'RGB is observed; points/skeleton are model forecasts. Visibility is not predicted.']
                right = [f'Point + FK | query {index} | +{offset / 20:.2f}s of 1.60s',
                         'Current anchors (new query)' if offset == 0 else
                         ('Fixed-query forecast; later replans may differ' if args.query_index is not None else
                          ('Adopted-query forecast; offline alignment' if adoption_steps is not None else
                           'Latest-query forecast; identities reset at each replan'))]
                for col, texts in enumerate([lines, right]):
                    for line, text in enumerate(texts):
                        cv2.putText(canvas, text, (col * width + 8, 24 + 25 * line),
                                    cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
                if rendered == 0:
                    cv2.imwrite(str(args.output.with_suffix('.jpg')), canvas)
                rendered += 1
        finally:
            process.stdin.close()
            return_code = process.wait()
        if return_code:
            raise RuntimeError('ffmpeg failed')
    report = dict(video=str(args.output), frames=rendered, fps=20 / args.record_stride,
                  queries=len(archives), query_index=args.query_index, source_episode=str(args.episode),
                  predictions=str(args.predictions), mode=('adopted_query' if adoption_steps is not None else 'latest_query') if args.query_index is None else 'fixed_query',
                  visibility='finite positive depth and image bounds only; not an occlusion estimate',
                  rgb='observed simulator RGB, not generated video',
                  trajectory='anchor + metric displacement; no accumulation, rescaling or smoothing')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
