import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from .interaction_validation import load_halves
from .predict import CONTROL_SECONDS, controlled
from .prepare import MATCH_IDS


def spell(truths, predictions):
    matched, total = np.zeros(len(pm.SPELL_IOUS)), 0
    for truth, labels in zip(truths, predictions, strict=True):
        scores = pm.spell_scores(truth, labels, pm.valid_runs(truth))
        total += scores['n_gt'] + scores['n_pred']
        matched += [scores['per_iou'][q]['n_matched'] for q in pm.SPELL_IOUS]
    return float(2 * matched.mean() / total)


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'work', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for condition in ('masked', 'original'):
        rows = {}
        for match in MATCH_IDS:
            truths = [half['truth'] for half in load_halves(args.cache, match, condition, ('truth',))]
            raw = [half['combined'] for half in load_halves(args.work / 'oof', match, condition, ('combined',))]
            merged = [controlled(labels) for labels in raw]
            # 参考标签按同一规则改写，门槛为零表示外部比赛的原始标签。
            for seconds in (0, 1.5, 2, 3):
                reference = [controlled(truth, seconds) for truth in truths] if seconds else truths
                rows.setdefault(f'reference {seconds:g} s, decoder output', []).append(spell(reference, raw))
                rows.setdefault(f'reference {seconds:g} s, merged below {CONTROL_SECONDS:g} s', []).append(spell(reference, merged))
        report[condition] = {name: {'mean': float(np.mean(values)), 'matches': dict(zip(MATCH_IDS, values, strict=True))}
                             for name, values in rows.items()}
        for name, values in rows.items():
            print(f'{condition}  {name}  {np.mean(values):.4f}', flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
