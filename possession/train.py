import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as pm
from starter.baseline import smooth
from .features import FPS, STRIDE
from .model import DIRECT_COSTS, decode_probabilities, fit_external, state_probabilities
from .prepare import CONDITIONS, MATCH_IDS


def load_games(directory):
    games = {}
    for match in MATCH_IDS:
        conditions = {}
        for condition in CONDITIONS:
            halves = []
            for period in (1, 2):
                with np.load(directory / f'{match}-{condition}-{period}.npz') as data:
                    halves.append({k: data[k] for k in data.files})
            conditions[condition] = halves
        games[match] = conditions
    return games


def training_samples(games, exclude=()):
    return [(half, half['truth'], step) for match, conditions in games.items() if match not in exclude
            for condition, step in zip(CONDITIONS, (3, 6, 3, 6), strict=True) for half in conditions[condition]]


def long_missing(visible):
    edges = np.diff(np.r_[False, ~visible, False].astype(np.int8))
    result = np.zeros(len(visible), dtype=bool)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if end - start > 30 * FPS:
            result[start:end] = True
    return result


def score(truth, pred, runs):
    spells = pm.spell_scores(truth, pm.fill_abstentions(pred, runs), runs)
    cm = pm.scored_confusion(pm.confusion(truth, pred))
    return {'spellF1': spells['mean_f1'], 'macroF1': pm.macro_f1(cm),
            'true_spells': spells['n_gt'], 'predicted_spells': spells['n_pred']}


def evaluate_game(halves, predictions):
    truth, predicted, baseline, runs = [], [], [], []
    offset = 0
    visible = hidden_alive = hidden_correct = shape_correct = 0
    for half, pred in zip(halves, predictions, strict=True):
        y = half['truth']
        truth.append(y)
        predicted.append(pred)
        baseline.append(pm.label_codes(smooth(half['nearest'], 50)))
        runs.extend((lo + offset, hi + offset) for lo, hi in pm.valid_runs(y))
        offset += len(y)
        visible += int(half['visible'].sum())
        hidden = long_missing(half['visible']) & (y < 2)
        hidden_alive += int(hidden.sum())
        hidden_correct += int((pred[hidden] == y[hidden]).sum())
        shape = np.repeat(half['shape_logit'] > 0, STRIDE)[:len(y)]
        shape_correct += int((shape[hidden] == y[hidden]).sum())
    truth = np.concatenate(truth)
    return {'frames': offset, 'graded_frames': int((truth != pm.X).sum()), 'visible_frames': visible,
            'model': score(truth, np.concatenate(predicted), runs),
            'smooth50': score(truth, np.concatenate(baseline), runs),
            'long_gap_alive': hidden_alive, 'long_gap_correct': hidden_correct,
            'long_gap_shape_correct': shape_correct}


def profile_scores(halves, probabilities, rates=None):
    results = []
    for costs in DIRECT_COSTS:
        matched = np.zeros(len(pm.SPELL_IOUS))
        total = 0
        for half, probs in zip(halves, probabilities, strict=True):
            y = half['truth']
            pred = decode_probabilities(probs, half['visible'], costs, rates)
            scores = pm.spell_scores(y, pred, pm.valid_runs(y))
            total += scores['n_gt'] + scores['n_pred']
            matched += [scores['per_iou'][q]['n_matched'] for q in pm.SPELL_IOUS]
        results.append(float(2 * matched.mean() / total))
    return results


def choose_costs(scores):
    return DIRECT_COSTS[int(np.mean(list(scores.values()), axis=0).argmax())]


def inner_validation(games):
    # 同时剔除外层测试比赛和内层验证比赛，一次拟合供两个方向复用。
    scores = {match: {} for match in games}
    for first, second in combinations(games, 2):
        models = fit_external(training_samples(games, (first, second)))
        for outer, inner in ((first, second), (second, first)):
            halves = games[inner]['masked']
            probs = [state_probabilities(models, half) for half in halves]
            scores[outer][inner] = profile_scores(halves, probs)
        print('内层留出', first, second, flush=True)
    return scores


def fit_with_validation(games, validation, oof_dir):
    inner = inner_validation(games) if validation else {}
    select_final, report = {}, {}
    for match, conditions in games.items():
        models = fit_external(training_samples(games, (match,)))
        masked = conditions['masked']
        masked_probs = [state_probabilities(models, half) for half in masked]
        select_final[match] = profile_scores(masked, masked_probs)
        if validation:
            costs = choose_costs(inner[match])
            result = {'direct_costs': costs, 'inner_scores': inner[match]}
            for condition, halves in conditions.items():
                probs = masked_probs if condition == 'masked' else [state_probabilities(models, h) for h in halves]
                predictions = []
                for period, (half, p) in enumerate(zip(halves, probs, strict=True), 1):
                    pred = decode_probabilities(p, half['visible'], costs)
                    predictions.append(pred)
                    if oof_dir:
                        raw = np.repeat(p.argmax(axis=1).astype(np.int8), STRIDE)[:len(pred)]
                        np.savez_compressed(oof_dir / f'{match}-{condition}-{period}.npz', pred=pred, raw=raw, probability=p)
                result[condition] = evaluate_game(halves, predictions)
            report[match] = result
            print(match, costs, result['masked']['model'], flush=True)
    if validation:
        validation.parent.mkdir(parents=True, exist_ok=True)
        validation.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    models = fit_external(training_samples(games))
    models['direct_costs'] = choose_costs(select_final)
    print('最终直接换队惩罚', models['direct_costs'], flush=True)
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--oof', type=Path, help='保存外层测试比赛的预测供逐段核查')
    args = parser.parse_args()
    if args.oof:
        if not args.validation:
            parser.error('--oof 需要同时指定 --validation')
        args.oof.mkdir(parents=True, exist_ok=True)
    games = load_games(args.cache)
    with threadpool_limits(limits=1):
        models = fit_with_validation(games, args.validation, args.oof)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    staged = args.model.with_suffix('.joblib.part')
    joblib.dump(models, staged, compress=3)
    staged.replace(args.model)


if __name__ == '__main__':
    main()
