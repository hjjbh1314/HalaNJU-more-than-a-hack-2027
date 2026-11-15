import argparse
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits

from .features import FPS, match_features
from .tracking import read_idsse, read_target


MATCH_IDS = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')
CONDITIONS = ('masked', 'original', 'hidden-masked', 'hidden')
SMOOTH_FRAMES = 7
STILL_SECONDS = 5


def observed_mask(reference, n, rng):
    offset = int(rng.integers(len(reference)))
    clock = (np.arange(n, dtype=np.int64) * len(reference) // n + offset) % len(reference)
    return reference[clock]


def runs(flag):
    change = np.flatnonzero(np.diff(np.r_[False, flag, False].astype(np.int8)))
    return zip(change[::2], change[1::2])


def smooth_ball(ball):
    # 目标比赛的球轨迹经过平滑，这里在每个连续可见段内做相同强度的高斯平滑。
    out = ball.copy()
    for start, end in runs(np.isfinite(ball).all(axis=1)):
        out[start:end] = gaussian_filter1d(ball[start:end].astype(np.float64), SMOOTH_FRAMES, axis=0, mode='nearest')
    return out


def still_ball(ball):
    # 外部数据在死球时把球固定在原地，目标比赛长时间静止的球基本缺失。
    step = np.r_[0, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    still = np.zeros(len(ball), dtype=bool)
    for start, end in runs(step < .5):
        if end - start >= STILL_SECONDS * FPS:
            still[start:end] = True
    return still


def prepare_match(data, directory, match_id, masks, interactions=False):
    halves, truths = read_idsse(data, match_id)
    rng = np.random.default_rng(zlib.crc32(match_id.encode('ascii')))
    conditions = {name: [] for name in CONDITIONS}
    for half, reference in zip(halves, masks, strict=True):
        replayed = ~observed_mask(reference, len(half.ball), rng)
        still = still_ball(half.ball)
        for name, hidden in zip(CONDITIONS, (replayed, np.zeros_like(still), replayed | still, still), strict=True):
            ball = half.ball.copy()
            ball[hidden] = np.nan
            conditions[name].append(replace(half, ball=smooth_ball(ball)))
    for condition, inputs in conditions.items():
        for period, (features, truth) in enumerate(zip(match_features(inputs, interactions), truths, strict=True), 1):
            if len(features['nearest']) != len(truth):
                raise ValueError('跟踪数据与外部参考标签长度不同')
            path = directory / f'{match_id}-{condition}-{period}.npz'
            with path.with_suffix('.npz.part').open('wb') as handle:
                np.savez_compressed(handle, **features, truth=truth)
            path.with_suffix('.npz.part').replace(path)
    counts = np.bincount(np.concatenate(truths), minlength=4)
    print(f'{match_id}  H/A/D/X {counts.tolist()}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idsse', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--matches', nargs='+', choices=MATCH_IDS, default=MATCH_IDS)
    parser.add_argument('--interactions', action='store_true', help='计算近球人数、接近速度和多人跑动特征')
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    _, masks, _, _, _ = read_target(args.data_dir)
    with threadpool_limits(limits=1):
        for match_id in args.matches:
            prepare_match(args.idsse, args.cache, match_id, masks, args.interactions)


if __name__ == '__main__':
    main()
