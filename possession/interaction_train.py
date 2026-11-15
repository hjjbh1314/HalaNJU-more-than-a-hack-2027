import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .model import combine_probabilities, fit_interactions, probability_components
from .prepare import MATCH_IDS
from .train import load_games, training_samples


EXPERTS = ('density', 'closing', 'movement', 'joint')
FEATURE_SETS = {'baseline': (), 'density': ('density',), 'closing': ('closing',),
                'movement': ('movement',), 'combined': ('joint', 'movement')}
VARIANTS = ('baseline', 'density', 'closing', 'elapsed', 'movement', 'combined')
GAMES = {}


def initialize(cache):
    GAMES.update(load_games(cache))


def fit_fold(games, exclude, groups):
    native = [h for match in games if match not in exclude for h in games[match]['original']]
    return fit_interactions(training_samples(games, exclude), native, groups)


def write_fold(task):
    exclude, folder = task
    path = folder / '-'.join(exclude)
    path.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        models = fit_fold(GAMES, exclude, EXPERTS)
        np.save(path / 'duration.npy', models['duration_rates'])
        for match in exclude:
            for condition in (('masked', 'original') if len(exclude) == 1 else ('masked',)):
                for period, half in enumerate(GAMES[match][condition], 1):
                    parts = probability_components(models, half, EXPERTS)
                    probabilities = {key: combine_probabilities(parts, half, True, groups)
                                     for key, groups in FEATURE_SETS.items()}
                    output = path / f'{match}-{condition}-{period}.npz'
                    with output.with_suffix('.part').open('wb') as handle:
                        np.savez_compressed(handle, **probabilities)
                    output.with_suffix('.part').replace(output)
    print('完成留出拟合', *exclude, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--folds', type=Path, help='生成七个外层及二十一个内层拟合的概率')
    modes.add_argument('--model', type=Path, help='用七场数据拟合交付模型')
    parser.add_argument('--calibration', type=Path, help='验证程序计算的最终换队惩罚')
    args = parser.parse_args()
    if args.folds:
        excluded = list(combinations(MATCH_IDS, 2)) + [(m,) for m in MATCH_IDS]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.cache,)) as pool:
            list(pool.map(write_fold, [(item, args.folds) for item in excluded]))
    else:
        if args.calibration is None:
            parser.error('--model 需要 --calibration')
        costs = json.loads(args.calibration.read_text())['final_costs']['combined']
        with threadpool_limits(limits=1):
            trained = fit_fold(load_games(args.cache), (), ('joint', 'movement'))
        trained['direct_costs'] = tuple(costs)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        staged = args.model.with_suffix('.joblib.part')
        joblib.dump(trained, staged, compress=3)
        staged.replace(args.model)
        print('已拟合七场比赛', args.model, costs, flush=True)


if __name__ == '__main__':
    main()
