import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FPS, STRIDE, centers
from .elapsed import duration_scores, elapsed_decode


DIRECT_COSTS = ((0.5, 2.0), (1.5, 3.5), (2.5, 5.0), (3.5, 6.5))


def team_matrix(features, indices=slice(None)):
    return np.column_stack((features['team'][indices], features['handover'][indices]))


def movement_matrix(features, group, indices):
    extra = []
    groups = ('density', 'closing') if group == 'joint' else (group,)
    for key in groups:
        data = features[key]
        extra.append(data[indices])
        if key != 'movement':
            clock = np.arange(len(data))[indices]
            extra.extend((data[np.maximum(clock - 2, 0)], data[np.minimum(clock + 2, len(data) - 1)]))
    return np.column_stack((team_matrix(features, indices), *extra))


def fit_owner(matrix, owners, weight):
    model = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=23, min_samples_leaf=45,
                                          l2_regularization=4, early_stopping=False, random_state=0)
    return model.fit(matrix, owners, sample_weight=weight / weight.mean())


def team_training_rows(truth, step):
    clock = centers(len(truth))
    bounds = np.r_[0, np.flatnonzero(np.diff(truth)) + 1, len(truth)]
    runs = np.searchsorted(bounds[1:], clock, side='right')
    start, end = bounds[runs], bounds[runs + 1]
    near_edge = np.minimum(clock - start, end - 1 - clock) < FPS
    sampled = (np.arange(len(clock)) % step == 0) | near_edge
    indices = np.flatnonzero(sampled & (truth[clock] < 2))
    # 保留边界附近全部采样点，用采样密度校正权重，再提高短段在损失中的比重。
    duration_weight = np.clip(np.sqrt(6 * FPS / (end - start)), 0.6, 3.5)
    weight = duration_weight * np.where(near_edge, 1 / step, 1)
    return indices, truth[clock[indices]], weight[indices]


def fit_external(training):
    labels, teams, stoppages, owners, weights = [], [], [], [], []
    for features, truth, step in training:
        indices = np.arange(0, len(features['team']), step)
        sample = truth[centers(len(truth))[indices]]
        keep = sample < 3
        labels.append(sample[keep])
        stoppages.append(features['dead'][indices[keep]])
        team_idx, owner, weight = team_training_rows(truth, step)
        teams.append(team_matrix(features, team_idx))
        owners.append(owner)
        weights.append(weight)
    y = np.concatenate(labels)
    options = dict(max_iter=150, max_leaf_nodes=15, min_samples_leaf=80,
                   l2_regularization=4, early_stopping=False, random_state=0)
    dead = HistGradientBoostingClassifier(**options).fit(np.concatenate(stoppages), y == 2)
    owner_weight = np.concatenate(weights)
    team = fit_owner(np.concatenate(teams), np.concatenate(owners), owner_weight)
    return dict(dead=dead, team=team, version=2)


def fit_interactions(training, native_halves, groups):
    models = fit_external(training)
    samples = [team_training_rows(truth, step) for _, truth, step in training]
    owners = np.concatenate([owner for _, owner, _ in samples])
    weights = np.concatenate([weight for _, _, weight in samples])
    experts = {}
    for group in groups:
        blocks = [movement_matrix(half, group, indices) for (half, _, _), (indices, _, _) in zip(training, samples, strict=True)]
        experts[group] = fit_owner(np.concatenate(blocks), owners, weights)
    models.update(version=3, experts=experts, duration_rates=duration_scores(native_halves))
    return models


TAG_WEIGHTS = {'BUILD UP': 1.4, 'PROGRESSION': 1.2, 'ATTACKING TRANSITION': 0.65,
               'BALL IN FINAL THIRD': 0.5, 'CREATING CHANCES': 0.45, 'LONG BALL': 0.4,
               'BALL IN THE BOX': 0.35, 'PLAYERS IN THE BOX': 0.15}


def tag_priors(n, start, names, events):
    clock = centers(n) + start
    team_weights = np.zeros((2, len(clock)))
    restart = np.zeros(len(clock))
    for event in events:
        code = event['code']
        begin, end = event['start_frame'], event['end_frame']
        if end < clock[0] - 10 * FPS or begin > clock[-1] + 10 * FPS:
            continue
        if code == 'SET PIECES':
            seconds = (clock - begin) / FPS
            bump = 1.05 * np.exp(-0.5 * ((seconds - 1) / 3.5) ** 2)
            restart = np.maximum(restart, bump)
        name = event['tags'].get('Team')
        if code in TAG_WEIGHTS and name in names:
            # 区间两端一秒内逐渐减弱权重。重叠区间每队只取最高值。
            taper = np.clip(np.minimum(clock - begin, end - clock) / FPS, 0, 1)
            side = names.index(name)
            team_weights[side] = np.maximum(team_weights[side], TAG_WEIGHTS[code] * taper)
    return team_weights[1] - team_weights[0], restart


def decode(emission, direct_cost=None):
    n = len(emission)
    duration = (3, 3, 5) if direct_cost is None else (1, 1, 5)
    direct = np.full(n, 3.5) if direct_cost is None else direct_cost
    prefix = np.vstack((np.zeros(3), np.cumsum(emission, axis=0)))
    value = np.full((n + 1, 3), -np.inf)
    previous = np.full((n + 1, 3), 3, dtype=np.int8)
    value[0] = 0
    for end in range(1, n + 1):
        for state, minimum in enumerate(duration):
            begin = end - minimum
            if begin < 0:
                continue
            left, right = (state + 1) % 3, (state + 2) % 3
            via_left = value[begin, left] - (direct[begin] if state < 2 and left < 2 else 3.5)
            via_right = value[begin, right] - (direct[begin] if state < 2 and right < 2 else 3.5)
            best = left if via_left >= via_right else right
            switch = (max(via_left, via_right) if begin else 0) + prefix[end, state] - prefix[begin, state]
            stay = value[end - 1, state] + emission[end - 1, state]
            if switch > stay:
                value[end, state] = switch
                previous[end, state] = best if begin else -1
            else:
                value[end, state] = stay
    if not np.isfinite(value[n]).any():
        raise ValueError('半场长度或概率不足以进行时序解码')
    state = int(value[n].argmax())
    labels = np.empty(n, dtype=np.int8)
    end = n
    while end:
        parent = int(previous[end, state])
        step = 1 if parent == 3 else duration[state]
        labels[end - step:end] = state
        end -= step
        if parent != 3:
            state = parent
    return labels


def probability_components(models, features, groups):
    current = models.get('version', 1) in (2, 3)
    team_inputs = team_matrix(features) if current else features['team']
    team_model = logit(np.clip(models['team'].predict_proba(team_inputs)[:, 1], 0.01, 0.99))
    dead_model = logit(np.clip(models['dead'].predict_proba(features['dead'])[:, 1], 0.005, 0.995))
    visible = features['visible'][centers(len(features['visible']))]
    updates = {}
    for group in groups:
        rows = np.flatnonzero(~visible if group == 'movement' else visible)
        correction = np.zeros_like(team_model)
        if rows.size:
            estimate = models['experts'][group].predict_proba(movement_matrix(features, group, rows))[:, 1]
            correction[rows] = .65 * (logit(np.clip(estimate, .01, .99)) - team_model[rows])
        updates[group] = correction
    return team_model, dead_model, updates


def combine_probabilities(parts, features, current, groups, tag_team=0, tag_dead=0):
    team_model, dead_model, updates = parts
    for group in groups:
        team_model = team_model + updates[group]
    external_weight = 0.86 if current else 0.60
    p_away = expit(external_weight * team_model + (1 - external_weight) * features['prior'] + tag_team)
    p_dead = expit(dead_model + tag_dead)
    return np.column_stack(((1 - p_dead) * (1 - p_away), (1 - p_dead) * p_away, p_dead))


def state_probabilities(models, features, tag_team=0, tag_dead=0):
    version = models.get('version', 1)
    if version not in (1, 2, 3):
        raise ValueError('模型版本未知')
    groups = ('joint', 'movement') if version == 3 else ()
    parts = probability_components(models, features, groups)
    return combine_probabilities(parts, features, version >= 2, groups, tag_team, tag_dead)


def decode_probabilities(probs, visible, costs, rates=None):
    direct = None
    if costs is not None:
        local_visibility = uniform_filter1d(visible.astype(float), size=FPS, mode='nearest')[::STRIDE]
        direct = costs[1] + (costs[0] - costs[1]) * local_visibility
    emission = np.log(probs)
    labels = decode(emission, direct) if rates is None else elapsed_decode(emission, direct, rates)
    return np.repeat(labels, STRIDE)[:len(visible)]


def predict_half(models, features, tag_team=0, tag_dead=0):
    probs = state_probabilities(models, features, tag_team, tag_dead)
    costs = models['direct_costs'] if models.get('version', 1) >= 2 else None
    return decode_probabilities(probs, features['visible'], costs, models.get('duration_rates'))
