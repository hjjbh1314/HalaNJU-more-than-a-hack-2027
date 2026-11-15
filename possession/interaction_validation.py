import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from .features import FPS
from .interaction_train import VARIANTS
from .model import DIRECT_COSTS, decode_probabilities
from .prepare import MATCH_IDS
from .train import choose_costs, evaluate_game, long_missing, profile_scores


def save_json(path, data):
    staged = path.with_suffix('.part')
    staged.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    staged.replace(path)


def load_halves(directory, match, condition, fields=None):
    halves = []
    for period in (1, 2):
        with np.load(directory / f'{match}-{condition}-{period}.npz', allow_pickle=False) as data:
            halves.append({k: data[k] for k in (data.files if fields is None else fields)})
    return halves


def get_probabilities(outputs, variant):
    key = 'baseline' if variant == 'elapsed' else variant
    return [h[key] for h in outputs]


def get_rates(rates, variant):
    return rates if variant in ('elapsed', 'combined') else None


def profile_fold(task):
    cache, folds, profiles, exclude = task
    name = '-'.join(exclude)
    result = {}
    rates = np.load(folds / name / 'duration.npy')
    for match in exclude:
        halves = load_halves(cache, match, 'masked', ('truth', 'visible'))
        outputs = load_halves(folds / name, match, 'masked')
        result[match] = {v: profile_scores(halves, get_probabilities(outputs, v), get_rates(rates, v))
                         for v in VARIANTS}
    save_json(profiles / f'{name}.json', result)
    print('完成惩罚验证', name, flush=True)


def audit_boundaries(halves, predictions):
    counts, events = Counter(), []
    for period, (half, labels) in enumerate(zip(halves, predictions, strict=True), 1):
        visible, truth = half['visible'], half['truth']
        long_gap = long_missing(visible)
        for lo, hi in pm.valid_runs(truth):
            run = [(lo, hi)]
            real = [t for t in pm.transitions(truth, run) if max(t[1:]) < pm.D]
            pred = [t for t in pm.transitions(labels, run) if max(t[1:]) < pm.D]
            pairs, _, _ = pm.match_transitions(real, pred, FPS, typed='pair')
            matched = dict(pairs)

            def ball_status(t):
                near = visible[max(lo, t[0] - FPS // 2):min(hi, t[0] + FPS // 2 + 1)]
                return 'visible' if near.all() else 'mixed' if near.any() else 'hidden'

            hidden_real = [t for t in real if ball_status(t) == 'hidden']
            hidden_pred = [t for t in pred if ball_status(t) == 'hidden']
            strict_pairs, _, strict_fp = pm.match_transitions(hidden_real, hidden_pred, FPS, typed='pair')
            counts.update(direct_ref=len(real), direct_pred=len(pred), direct_hit=len(pairs),
                          hidden_ref=len(hidden_real), hidden_pred=len(hidden_pred), hidden_strict_hit=len(strict_pairs),
                          hidden_strict_fp=len(strict_fp))
            for t in real:
                status = ball_status(t)
                counts[f'{status}_reference'] += 1
                counts[f'{status}_hit'] += int(t in matched)
                if long_gap[t[0]]:
                    counts['long_gap_ref'] += 1
                    counts['long_gap_hit'] += int(t in matched)
                events.append((period, *t, status, int(long_gap[t[0]]), matched[t][0] if t in matched else None))
    return dict(counts), events


def evaluate_outer(task):
    cache, folds, work, match, inner, outer_profile = task
    directory = folds / match
    rates = np.load(directory / 'duration.npy')
    costs = {v: choose_costs({other: scores[v] for other, scores in inner.items()}) for v in VARIANTS}
    reference_cost = DIRECT_COSTS.index(costs['baseline'])
    result = {v: {'direct_costs': costs[v], 'fixed_cost_masked': outer_profile[v][reference_cost]}
              for v in VARIANTS}
    events = {}
    for condition in ('masked', 'original'):
        halves = load_halves(cache, match, condition, ('truth', 'nearest', 'visible', 'shape_logit'))
        outputs = load_halves(directory, match, condition)
        predictions = {}
        for variant in VARIANTS:
            probs = get_probabilities(outputs, variant)
            predictions[variant] = [decode_probabilities(p, h['visible'], costs[variant], get_rates(rates, variant))
                                    for p, h in zip(probs, halves, strict=True)]
            measures = evaluate_game(halves, predictions[variant])
            result[variant][condition] = measures
            if condition == 'masked':
                index = DIRECT_COSTS.index(costs[variant])
                np.testing.assert_allclose(measures['model']['spellF1'], outer_profile[variant][index], atol=1e-14)
                result[variant]['transitions'], events[variant] = audit_boundaries(halves, predictions[variant])
        for i in (0, 1):
            path = work / 'oof' / f'{match}-{condition}-{i + 1}.npz'
            with path.with_suffix('.part').open('wb') as handle:
                np.savez_compressed(handle, **{v: arrays[i] for v, arrays in predictions.items()})
            path.with_suffix('.part').replace(path)
    rows = []
    for bundle in zip(*(events[v] for v in VARIANTS), strict=True):
        period, frame, source, dest, status, long_gap, _ = bundle[0]
        if any(event[:-1] != bundle[0][:-1] for event in bundle):
            raise ValueError('不同模型的真实换队记录不一致')
        rows.append((match, period, frame, pm.LABELS[source], pm.LABELS[dest], status, long_gap,
                     *(event[-1] for event in bundle)))
    return match, result, rows


def validate(cache, folds, work, report, workers, available_only, reference):
    profiles = work / 'profiles'
    profiles.mkdir(parents=True, exist_ok=True)
    exclusions = list(combinations(MATCH_IDS, 2)) + [(m,) for m in MATCH_IDS]
    tasks = []
    for exclude in exclusions:
        name = '-'.join(exclude)
        condition = 'masked' if len(exclude) == 2 else 'original'
        done = folds / name / f'{exclude[-1]}-{condition}-2.npz'
        if done.exists():
            if not available_only or not (profiles / f'{name}.json').exists():
                tasks.append((cache, folds, profiles, exclude))
        elif not available_only:
            raise FileNotFoundError(done)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(profile_fold, tasks))
    if available_only:
        return
    (work / 'oof').mkdir(exist_ok=True)
    scores = {exclude: json.loads((profiles / ('-'.join(exclude) + '.json')).read_text()) for exclude in exclusions}
    inner = {m: {n: scores[tuple(sorted((m, n)))][n] for n in MATCH_IDS if m != n} for m in MATCH_IDS}
    tasks = [(cache, folds, work, m, inner[m], scores[(m,)][m]) for m in MATCH_IDS]
    measures, events = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for match, result, rows in pool.map(evaluate_outer, tasks):
            measures[match] = result
            events.extend(rows)
    if reference:
        previous = json.loads(reference.read_text())
        for m in MATCH_IDS:
            base = measures[m]['baseline']
            for key in ('masked', 'original'):
                if base[key] != previous[m][key]:
                    raise ValueError(f'{m} 的 {key} 基准未复现')
            np.testing.assert_equal(base['direct_costs'], previous[m]['direct_costs'])
            for n in inner[m]:
                np.testing.assert_allclose(inner[m][n]['baseline'], previous[m]['inner_scores'][n], atol=1e-14)
    with (work / 'direct_transitions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('match', 'half', 'frame', 'source', 'dest', 'ball', 'long_gap', *VARIANTS))
        writer.writerows(events)
    final_costs = {v: choose_costs({m: scores[(m,)][m][v] for m in MATCH_IDS}) for v in VARIANTS}
    averages = {v: {k: float(np.mean([measures[m][v][k]['model']['spellF1'] for m in MATCH_IDS]))
                    for k in ('masked', 'original')} for v in VARIANTS}
    report.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, {'folds': measures, 'means': averages, 'inner_profiles': inner,
                       'outer_profiles': {m: scores[(m,)][m] for m in MATCH_IDS}, 'final_costs': final_costs})
    print(json.dumps(averages, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'folds', 'work', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--available-only', action='store_true')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    validate(args.cache, args.folds, args.work, args.report, args.workers, args.available_only, args.reference)


if __name__ == '__main__':
    main()
