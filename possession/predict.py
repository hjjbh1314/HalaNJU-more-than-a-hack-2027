import argparse
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as pm
from games import DATA_DIR, GAME, half_video_blocks, source_frames
from .features import FPS, match_features
from .model import predict_half, tag_priors
from .tracking import read_target


CONTROL_SECONDS = 1.5


def controlled(labels, seconds=CONTROL_SECONDS):
    # 官方只把有控制的控球算作换队：由直接换队得到且不足门槛时长的控球段并回前一支球队。
    out = labels.copy()
    bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        team = labels[start]
        if team < 2 and start > 0 and out[start - 1] == 1 - team and end - start < seconds * FPS:
            out[start:end] = 1 - team
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(DATA_DIR))
    parser.add_argument('--model', type=Path, default=Path(__file__).with_name('idsse_models.joblib'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models = joblib.load(args.model)
    halves, _, starts, names, events = read_target(args.data_dir)
    labels = np.full(source_frames(), pm.X, dtype=np.int8)
    with threadpool_limits(limits=1):
        features_by_half = match_features(halves, interactions=models.get('version') == 3)
        for features, kickoff, (_, begin, end) in zip(features_by_half, starts, half_video_blocks(GAME), strict=True):
            n = end + 1 - kickoff
            priors = tag_priors(n, kickoff, names, events)
            labels[begin:kickoff] = pm.D
            labels[kickoff:end + 1] = controlled(predict_half(models, features, *priors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=args.output.parent, suffix='.csv.part', encoding='utf-8',
                            newline='', delete=False) as handle:
        staged = Path(handle.name)
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('frame', 'label'))
        writer.writerows(enumerate(np.asarray(pm.LABELS)[labels]))
    error = pm.validate_submission(pm.read_label_csv(staged), source_frames())
    if error:
        raise ValueError(error)
    staged.replace(args.output)
    print(f'{args.output}  {len(labels):,} frames', flush=True)


if __name__ == '__main__':
    main()
