import numpy as np

from .features import centers


AGE_BINS = np.array([1, 3, 5, 10, 20, 40, 80])


def duration_scores(training_halves):
    counts = np.zeros((len(AGE_BINS), 3), dtype=np.float64)
    for half in training_halves:
        truth = half['truth']
        seq = truth[centers(len(truth))]
        boundaries = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            state = seq[start]
            if state > 1 or start == 0:
                continue
            length = end - start
            # 段内每个采样点贡献一次风险暴露，半场末端只计已观察到的延续。
            ages = np.arange(1, length)
            bands = np.searchsorted(AGE_BINS, ages, side='right') - 1
            counts[:, 0] += np.bincount(bands, minlength=len(AGE_BINS))
            if end < len(seq) and seq[end] < 3:
                band = np.searchsorted(AGE_BINS, length, side='right') - 1
                counts[band, 1 if seq[end] < 2 else 2] += 1
    marginal = counts.sum(axis=0) / counts.sum()
    if not np.all(marginal > 0):
        raise ValueError('训练比赛缺少控球延续、直接换队或死球记录')
    conditional = (counts + 200 * marginal) / (counts.sum(axis=1, keepdims=True) + 200)
    bands = np.searchsorted(AGE_BINS, np.arange(1, AGE_BINS[-1] + 1), side='right') - 1
    return .65 * np.clip(np.log(conditional[bands] / marginal), -1.5, 1.5)


def elapsed_decode(emission, direct_cost, rates):
    cap = len(rates)
    dead = 2 * cap
    states = dead + 5
    n = len(emission)
    score = np.full(states, -np.inf)
    score[[0, cap, dead]] = emission[0]
    parent = np.broadcast_to(np.arange(states, dtype=np.int16) - 1, (n, states)).copy()
    staying, turning, stopping = rates.T
    for t in range(1, n):
        live = score[:dead].reshape(2, cap)
        continued = live + staying
        transfer = live + turning
        turns = transfer.argmax(axis=1)
        stopped = live + stopping
        last_live = int(stopped.argmax())
        nxt = np.empty_like(score)
        live_next = nxt[:dead].reshape(2, cap)
        live_next[:, 1:] = continued[:, :-1]
        for side, other in ((0, 1), (1, 0)):
            begin, tail = side * cap, side * cap + cap - 1
            if continued[side, -1] >= continued[side, -2]:
                live_next[side, -1] = continued[side, -1]
                parent[t, tail] = tail
            came_from = other * cap + turns[other]
            switch = transfer[other, turns[other]] - direct_cost[t]
            restart = score[-1] - 3.5
            if restart > switch:
                came_from, switch = states - 1, restart
            nxt[begin] = switch
            parent[t, begin] = came_from
            live_next[side] += emission[t, side]
        nxt[dead] = stopped.flat[last_live] - 3.5
        parent[t, dead] = last_live
        nxt[dead + 1:] = score[dead:-1]
        if score[-1] >= score[-2]:
            nxt[-1] = score[-1]
            parent[t, -1] = states - 1
        nxt[dead:] += emission[t, 2]
        score = nxt
    # 死球至少持续五个采样点。球队的年龄随候选路径更新，两队互换时归零重计。
    score[dead:-1] = -np.inf
    state = int(score.argmax())
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = min(state // cap, 2)
        state = parent[t, state]
    return out
