from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

from .contacts import team_context, touch_context
from .player_path import carrier_examples, carrier_path, player_scene
from .pressure import ball_neighbours


FPS = 25
STRIDE = 5


def centers(n):
    return np.minimum(np.arange(0, n, STRIDE) + STRIDE // 2, n - 1)


def smooth(values, window):
    return pd.DataFrame(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def row_mean(values):
    count = np.isfinite(values).sum(axis=1)
    return np.nansum(values, axis=1) / np.where(count > 0, count, np.nan)


def row_quantile(values, fraction):
    count = np.isfinite(values).sum(axis=1)
    ranks = np.maximum(0, np.rint((count - 1) * fraction).astype(int))
    return np.sort(values, axis=1)[np.arange(len(values)), ranks]


def player_features(half, index):
    points = half.players[index] - half.pitch / 2
    left = np.maximum(index - 10, 0)
    right = np.minimum(index + 10, len(half.ball) - 1)
    velocity = (half.players[right] - half.players[left]) * (FPS / (right - left))[:, None, None]
    speed = np.linalg.norm(velocity, axis=2)
    speed[speed > 12] = np.nan
    home = half.sides == 0
    direction = np.sign(np.nanmean(points[:, ~home, 0]) - np.nanmean(points[:, home, 0]))
    if direction not in (-1, 1):
        raise ValueError('两队站位不足以确定进攻方向')
    points[:, :, 0] *= direction
    teams = []
    for side in (home, ~home):
        x, y = points[:, side, 0], points[:, side, 1]
        x10, x90 = row_quantile(x, 0.1), row_quantile(x, 0.9)
        width = row_quantile(y, 0.9) - row_quantile(y, 0.1)
        teams.append(np.column_stack((row_mean(x), row_mean(y), x10, x90, x90 - x10, width,
                                      row_mean(speed[:, side]), row_quantile(speed[:, side], 0.8),
                                      np.isfinite(x).sum(axis=1))))
    h, a = (smooth(team, 5) for team in teams)
    centre = np.column_stack((h[:, :2], a[:, :2]))
    centre_change = pd.DataFrame(centre).diff(5).to_numpy()
    shape = np.column_stack((h, a, h[:, :8] - a[:, :8], centre_change))
    pace = row_mean(speed)
    fast = row_mean(np.where(np.isfinite(speed), speed > 3, np.nan))
    pace_features = np.column_stack((pace, row_quantile(speed, 0.8), fast,
                                     pd.Series(pace).rank(pct=True), h[:, 5] + a[:, 5],
                                     np.abs(h[:, 0] - a[:, 0]), np.abs(h[:, 1] - a[:, 1])))
    return shape, pace_features, centre, direction


@dataclass
class HalfFeatures:
    shape: np.ndarray
    pseudo: np.ndarray
    touch: np.ndarray
    dead: np.ndarray
    nearest: np.ndarray
    visible: np.ndarray
    physical: np.ndarray


def extract(half):
    ball = half.ball
    n = len(ball)
    index = centers(n)
    shape, pace, centre, direction = player_features(half, index)
    visible = np.isfinite(ball).all(axis=1)
    distance = np.linalg.norm(half.players - ball[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    h, a = (distance[:, half.sides == s].min(axis=1) for s in (0, 1))
    nearest = np.where(np.isfinite(np.minimum(h, a)), (a < h).astype(np.int8), -1)
    near = np.minimum(h, a)
    near[~np.isfinite(near)] = np.nan
    h[~np.isfinite(h)] = np.nan
    a[~np.isfinite(a)] = np.nan
    margin = h - a
    edge = np.minimum(ball, half.pitch - ball).min(axis=1)
    step = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
    speed = pd.Series(step).rolling(5, center=True, min_periods=1).median().to_numpy(copy=True)
    speed[~visible] = np.nan

    # 一秒窗口内至少十帧满足近球条件，且对手没有满足条件的帧。
    candidate = (edge >= 0) & (near <= 1.5) & (np.abs(margin) >= 1) & (speed <= 8)
    votes = np.column_stack((candidate & (margin < 0), candidate & (margin > 0)))
    counts = pd.DataFrame(votes).rolling(FPS, center=True, min_periods=1).sum().to_numpy()
    pseudo = np.select([(counts[:, 0] >= 10) & (counts[:, 1] == 0),
                        (counts[:, 1] >= 10) & (counts[:, 0] == 0)], [0, 1], default=-1)

    # 触球要求一方更近，并且球速较低或来球与出球的速度向量有明显变化。
    impulse = np.full(n, np.nan)
    impulse[3:-3] = np.linalg.norm(ball[6:] - 2 * ball[3:-3] + ball[:-6], axis=1) * FPS / 3
    contiguous = pd.Series(visible).rolling(7, center=True, min_periods=7).sum().to_numpy() == 7
    contact = ((near <= 2.5) & (edge >= -0.5) & (np.abs(margin) >= 0.7)
               & ((speed <= 8) | (contiguous & (impulse >= 3))))
    physical = touch_context(ball, index, h, a, nearest, contact, np.where(contiguous, impulse, np.nan),
                             direction, FPS)
    frame = np.arange(n)
    previous = np.maximum.accumulate(np.where(contact, frame, -1))
    age = np.where(previous >= 0, (frame - previous) / FPS, np.inf)
    previous_sign = np.sign(margin[np.maximum(previous, 0)])
    carry = np.where(age <= 8, previous_sign * np.exp(-age / 2), 0)
    proximity = np.nan_to_num(np.clip(margin / 1.5, -3, 3) * np.exp(-near / 3))
    ball_centre = ball[index] - half.pitch / 2
    ball_centre[:, 0] *= direction
    relative = np.column_stack((ball_centre - centre[:, :2], ball_centre - centre[:, 2:]))
    touch = np.column_stack((carry[index], smooth(proximity, 5)[index, 0],
                             np.minimum(age[index], 12), h[index], a[index], speed[index],
                             ball_centre[:, 0], relative))

    ball_features = np.column_stack((np.minimum(speed[index], 40), edge[index], near[index],
                                     np.abs(ball_centre), visible[index]))
    current = np.column_stack((pace, ball_features))
    context = np.column_stack((pace[:, :4], ball_features[:, [0, 1, 2, 5]]))
    wide = smooth(context, 25)
    past = pd.DataFrame(context).rolling(15, min_periods=1).mean().to_numpy()
    future = pd.DataFrame(context[::-1]).rolling(15, min_periods=1).mean().to_numpy()[::-1]
    dead = np.column_stack((current, wide, past, future))
    return HalfFeatures(shape, pseudo[index], touch, dead, nearest, visible, physical)


def match_features(halves, interactions=False):
    features = [extract(half) for half in halves]
    result = []
    for test, train in zip(features, reversed(features), strict=True):
        has_label = train.pseudo >= 0
        if not np.array_equal(np.unique(train.pseudo[has_label]), [0, 1]):
            raise ValueError('另一半场的控球伪标签缺少球队')
        model = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, min_samples_leaf=30,
                                               l2_regularization=3, early_stopping=False, random_state=0)
        model.fit(train.shape[has_label], train.pseudo[has_label])
        shape_logit = logit(np.clip(model.predict_proba(test.shape)[:, 1], 0.02, 0.98))
        shape_logit = smooth(shape_logit, 5)[:, 0]
        team = np.column_stack((shape_logit, test.touch, test.shape))
        prior = 0.8 * shape_logit + 2.6 * test.touch[:, 0] + test.touch[:, 1]
        result.append(dict(team=team.astype(np.float32), dead=test.dead.astype(np.float32),
                           handover=team_context(shape_logit, test.touch, test.physical, test.shape),
                           prior=prior, nearest=test.nearest, visible=test.visible,
                           shape_logit=shape_logit))
    if interactions:
        scenes, examples = [], []
        for half, feature, output in zip(halves, features, result, strict=True):
            clock = centers(len(half.ball))
            output['density'], output['closing'] = ball_neighbours(half, clock, FPS)
            scene = player_scene(half, clock, FPS)
            scenes.append(scene)
            examples.append(carrier_examples(scene, half.ball[clock], feature.pseudo))
        for scene, sample, output in zip(scenes, reversed(examples), result, strict=True):
            output['movement'] = carrier_path(scene, sample)
    return result
