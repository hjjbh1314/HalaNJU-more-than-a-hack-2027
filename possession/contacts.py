import numpy as np
import pandas as pd


def touch_context(ball, index, home, away, near_team, contact, impulse, direction, fps):
    n = len(ball)
    frame = np.arange(n)
    binned_min = pd.DataFrame({'home': home, 'away': away}).rolling(5, center=True, min_periods=1).min()
    result = [binned_min.to_numpy()[index]]
    for side in (0, 1):
        found = contact & (near_team == side)
        previous = np.maximum.accumulate(np.where(found, frame, -n))
        following = np.minimum.accumulate(np.where(found, frame, 2 * n)[::-1])[::-1]
        result.extend((np.minimum((index - previous[index]) / fps, 8),
                       np.minimum((following[index] - index) / fps, 8)))
    # 同一短时间内的入球、出球方向和速度变化能保留瞬间触球的证据。
    left = np.maximum(index - 4, 0)
    right = np.minimum(index + 4, n - 1)
    vin = (ball[index] - ball[left]) * (fps / np.maximum(index - left, 1))[:, None]
    vout = (ball[right] - ball[index]) * (fps / np.maximum(right - index, 1))[:, None]
    valid = np.isfinite(ball).all(axis=1)
    missing = np.r_[0, np.cumsum(~valid)]
    vin[missing[index + 1] != missing[left]] = np.nan
    vout[missing[right + 1] != missing[index]] = np.nan
    for velocity in (vin, vout):
        result.extend((np.clip(velocity[:, 0] * direction, -40, 40),
                       np.minimum(np.linalg.norm(velocity, axis=1), 40)))
    result.append(np.minimum(np.linalg.norm(vout - vin, axis=1), 50))
    for distance in (home, away):
        strength = np.minimum(impulse, 40) * np.exp(-distance / 2)
        peak = pd.Series(strength).rolling(5, center=True, min_periods=1).max().to_numpy()
        result.append(peak[index])
    return np.column_stack(result)


def team_context(shape_logit, touch, physical, shape):
    current = pd.DataFrame(np.column_stack((shape_logit, touch[:, 1], touch[:, 3] - touch[:, 4],
                                            touch[:, 5], physical[:, 6], physical[:, 8])))
    matrix = [physical]
    # 各半场分别处理。缺球区间仍可观察两队前后移动和阵形变化。
    formation = pd.DataFrame(np.column_stack((shape_logit, shape[:, [0, 1, 5, 6, 9, 10, 14, 15]])))
    for width in (3, 8):
        matrix.extend((current.rolling(width, min_periods=1).mean().to_numpy(),
                       current.iloc[::-1].rolling(width, min_periods=1).mean().iloc[::-1].to_numpy()))
    for lag in (5, 15):
        matrix.extend((formation.diff(lag).to_numpy(), (formation.shift(-lag) - formation).to_numpy()))
    return np.column_stack(matrix).astype(np.float32)
