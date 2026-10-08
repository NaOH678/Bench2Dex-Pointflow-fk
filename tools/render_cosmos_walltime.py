"""Replay recorded RGB/overlay frames on measured wall time, including inference waits."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def timeline(timing, fps):
    if fps <= 0 or timing['record_stride'] != 1:
        raise ValueError('Positive FPS and recording stride=1 required')
    stamps = np.array([f['captured_wall_sec'] for f in timing['frames']])
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError('Missing or non-increasing capture timestamps')
    duration = timing['end_wall_sec'] - stamps[0]
    if duration <= 0 or timing['end_wall_sec'] < stamps[-1]:
        raise ValueError('Invalid end time')
    clock = stamps[0] + np.arange(int(np.ceil(duration * fps))) / fps
    indices = np.searchsorted(stamps, clock, side='right') - 1
    return clock, indices


def phase_at(timing, elapsed):
    for q in timing['queries']:
        if q['geometry_start_sec'] <= elapsed < q['geometry_end_sec']:
            return 'WAIT: POINT + FK PREPARATION'
        if q['geometry_end_sec'] <= elapsed < q['rpc_end_sec']:
            if not timing.get('async_inference', False):
                return 'WAIT: MODEL + RPC (PHYSICS PAUSED)'
            # Older logs store duration, not the exact future.result() start.
            # Reconstruct the blocking interval only for a measurable wait.
            wait = q.get('queue_wait_sec', 0) or 0
            if wait > .001 and elapsed >= q['rpc_end_sec'] - wait:
                return 'WAIT: EMPTY ACTION QUEUE / MODEL + RPC'
            return 'ASYNC MODEL + RPC / SIM CONTINUES'
    return 'SIM / CAMERA / RECORDING'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True, help='One frame per recorded control step; RGB or prediction overlay')
    p.add_argument('--timing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=20)
    a = p.parse_args()
    timing = json.loads(a.timing.read_text())
    clock, indices = timeline(timing, a.fps)
    cap = cv2.VideoCapture(str(a.video))
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(timing['frames']):
        raise ValueError('Video and timing frame counts differ')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'bgr24',
        '-video_size', f'{width}x{height + 72}', '-framerate', str(a.fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(a.output)], stdin=subprocess.PIPE)
    source_index = -1
    try:
        for elapsed, index in zip(clock, indices):
            while source_index < index:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Video decode failed')
                source_index += 1
            canvas = np.zeros((height + 72, width, 3), np.uint8)
            canvas[72:] = frame
            lines = [f"Wall {elapsed-clock[0]:.2f}s | Sim {timing['frames'][index]['sim_time_sec']:.2f}s",
                     phase_at(timing, elapsed)]
            for line, text in enumerate(lines):
                cv2.putText(canvas, text, (8, 27 + 30 * line), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 220, 255), 1, cv2.LINE_AA)
            proc.stdin.write(canvas.tobytes())
    finally:
        cap.release()
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError('ffmpeg failed')
    report = dict(video=str(a.output), frames=len(indices), fps=a.fps, seconds=len(indices)/a.fps,
        source_frames=len(timing['frames']), measured_seconds=timing['end_wall_sec']-clock[0],
        scope='first observation to control-loop end; startup and final encoding excluded',
        method='hold latest captured frame using measured monotonic timestamps, no optical interpolation')
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
