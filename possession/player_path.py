from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier

from .pressure import velocity_between


@dataclass
class PlayerScene:
    position: np.ndarray
    sides: np.ndarray
    features: np.ndarray


def local_relations(position, speed, sides):
    result = np.empty((*position.shape[:2], 12), dtype=np.float32)
    same = sides[:, None] == sides[None, :]
    other = ~same
    same = same & ~np.eye(len(sides), dtype=bool)
    for start in range(0, len(position), 128):
        part = slice(start, start + 128)
        pos, vel = position[part], speed[part]
        delta = pos[:, None, :, :] - pos[:, :, None, :]
        distances = np.linalg.norm(delta, axis=-1)
        relative = vel[:, :, None, :] - vel[:, None, :, :]
        approach = np.sum(relative * delta, axis=-1) / np.maximum(distances, .1)
        columns = []
        for eligible in (same, other):
            ds = np.where(eligible & np.isfinite(distances), distances, np.inf)
            nearest = ds.argmin(axis=2)[:, :, None]
            first_two = np.partition(ds, 1, axis=2)[:, :, :2]
            first = np.take_along_axis(ds, nearest, axis=2)[:, :, 0]
            nearest_approach = np.take_along_axis(approach, nearest, axis=2)[:, :, 0]
            second = first_two.max(axis=2)
            nearby = (ds < 12) & np.isfinite(approach)
            radial = np.where(nearby, approach, 0).sum(axis=2) / np.maximum(nearby.sum(axis=2), 1)
            columns.extend((np.where(np.isfinite(first), first, np.nan),
                            np.where(np.isfinite(second), second, np.nan),
                            (ds < 4).sum(axis=2), (ds < 10).sum(axis=2),
                            np.where(np.isfinite(first), nearest_approach, np.nan), radial))
        result[part] = np.stack(columns, axis=2)
    return result


def player_scene(half, clock, fps):
    pos = half.players[clock]
    valid = np.isfinite(pos).all(axis=2)
    centers = []
    for side in (0, 1):
        selected = half.sides == side
        centers.append(np.nansum(pos[:, selected], axis=1) /
                       np.maximum(valid[:, selected].sum(axis=1), 1)[:, None])
    direction = np.sign(np.nanmean(pos[:, half.sides == 1, 0]) - np.nanmean(pos[:, half.sides == 0, 0]))
    attacking = direction * np.where(half.sides == 0, 1, -1)
    own, opponent = (np.stack(centers, axis=1)[:, group] for group in (half.sides, 1 - half.sides))
    columns = [(pos[:, :, 0] - half.pitch[0] / 2) * attacking, np.abs(pos[:, :, 1] - half.pitch[1] / 2),
               (pos[:, :, 0] - own[:, :, 0]) * attacking, (pos[:, :, 0] - opponent[:, :, 0]) * attacking,
               np.abs(pos[:, :, 1] - own[:, :, 1]), np.abs(pos[:, :, 1] - opponent[:, :, 1])]
    for width in (10, 30):
        lo, hi = np.maximum(clock - width, 0), np.minimum(clock + width, len(half.players) - 1)
        # 速度窗口在半场边界截断。没有足够时间计算的一侧保持缺失。
        prev = velocity_between(half.players, lo, np.maximum(clock, lo + 1), fps, 12)
        after = velocity_between(half.players, np.minimum(clock, hi - 1), hi, fps, 12)
        prev[clock == lo] = np.nan
        after[clock == hi] = np.nan
        for velocity in (prev, after):
            columns.extend((velocity[:, :, 0] * attacking, np.abs(velocity[:, :, 1]),
                            np.linalg.norm(velocity, axis=2)))
        columns.append(np.linalg.norm(after - prev, axis=2))
    vel = velocity_between(half.players, np.maximum(clock - 8, 0), np.minimum(clock + 8, len(half.players) - 1), fps, 12)
    relations = local_relations(pos, vel, half.sides)
    matrix = np.concatenate((np.stack(columns, axis=2), relations), axis=2).astype(np.float32)
    return PlayerScene(pos, half.sides, matrix)


def carrier_examples(scene, ball, pseudo):
    dist = np.linalg.norm(scene.position - ball[:, None, :], axis=2)
    ds = np.where(np.isfinite(dist), dist, np.inf)
    nearest = ds.argmin(axis=1)
    chosen = ((pseudo == scene.sides[nearest]) & (ds.min(axis=1) < 2.0) & (np.arange(len(ds)) % 2 == 0))
    valid = np.isfinite(scene.position[chosen]).all(axis=2)
    labels = nearest[chosen, None] == np.arange(len(scene.sides))
    if chosen.sum() < 20:
        raise ValueError('另一半场的自动触球样本不足以拟合持球球员')
    return scene.features[chosen][valid], labels[valid]


def movement_edges(scene):
    pos, sides = scene.position, scene.sides
    n = len(sides)
    same = sides[:, None] == sides[None, :]
    edges = np.empty((len(pos) - 1, n + 1, n + 1), dtype=np.float32)
    ids = np.arange(n)
    for start in range(0, len(edges), 192):
        stop = min(start + 192, len(edges))
        previous, current = pos[start:stop], pos[start + 1:stop + 1]
        d = np.linalg.norm(previous[:, :, None] - current[:, None, :], axis=3)
        d = np.where(np.isfinite(d), d, np.inf)
        block = edges[start:stop]
        block[:, :n, :n] = np.exp(-d / np.where(same, 10., 4.)) * np.where(same, .18, .12)
        block[:, ids, ids] = 2.5 * np.exp(-d[:, ids, ids] / 3)
        block[:, :n, n] = .06
        block[:, n, :n] = np.isfinite(current[:, :, 0]) * .04
        block[:, n, n] = .5
        block /= block.sum(axis=2, keepdims=True)
    return edges


def carrier_path(scene, examples):
    x, y = examples
    model = HistGradientBoostingClassifier(max_iter=75, max_leaf_nodes=15, min_samples_leaf=45,
                                          l2_regularization=5, early_stopping=False, random_state=0)
    model.fit(x, y)
    valid = np.isfinite(scene.position).all(axis=2)
    evidence = np.zeros(valid.shape, dtype=np.float32)
    odds = logit(np.clip(model.predict_proba(scene.features[valid])[:, 1], .001, .95))
    evidence[valid] = np.exp(.45 * odds)
    likelihood = np.column_stack((evidence, np.full(len(valid), .18, dtype=np.float32)))
    edges = movement_edges(scene)
    forward, backward = np.empty_like(likelihood), np.ones_like(likelihood)
    forward[0] = likelihood[0] / likelihood[0].sum()
    for t in range(1, len(valid)):
        f = (forward[t - 1] @ edges[t - 1]) * likelihood[t]
        forward[t] = f / f.sum()
    for t in range(len(valid) - 2, -1, -1):
        b = edges[t] @ (likelihood[t + 1] * backward[t + 1])
        backward[t] = b / b.max()
    marginal = forward * backward
    marginal /= marginal.sum(axis=1, keepdims=True)
    home, away = (np.flatnonzero(scene.sides == side) for side in (0, 1))
    home_mass, away_mass = marginal[:, home].sum(axis=1), marginal[:, away].sum(axis=1)
    side_odds = np.log((away_mass + 1e-5) / (home_mass + 1e-5))
    turns = np.zeros((len(valid), 2), dtype=np.float32)
    for start in range(0, len(edges), 192):
        stop = min(start + 192, len(edges))
        pair = (forward[start:stop, :, None] * edges[start:stop] *
                (likelihood[start + 1:stop + 1] * backward[start + 1:stop + 1])[:, None, :])
        norm = pair.sum(axis=(1, 2))
        for j, (a, b) in enumerate(((home, away), (away, home))):
            turns[start + 1:stop + 1, j] = pair[:, a][:, :, b].sum(axis=(1, 2)) / norm
    strength = np.column_stack((home_mass, away_mass, marginal[:, -1], side_odds,
                                marginal[:, home].max(axis=1), marginal[:, away].max(axis=1), turns))
    return np.column_stack((strength, uniform_filter1d(strength, 7, axis=0, mode='nearest'),
                            uniform_filter1d(strength, 17, axis=0, mode='nearest'))).astype(np.float32)
