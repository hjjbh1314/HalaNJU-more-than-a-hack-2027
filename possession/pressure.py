import numpy as np


def velocity_between(points, start, stop, fps, limit):
    elapsed = (stop - start).astype(np.float32) / fps
    scale = elapsed.reshape((-1,) + (1,) * (points.ndim - 1))
    velocity = (points[stop] - points[start]) / scale
    velocity[np.linalg.norm(velocity, axis=-1) > limit] = np.nan
    return velocity


def ball_neighbours(half, clock, fps):
    players, ball = half.players[clock], half.ball[clock]
    offset = ball[:, None, :] - players
    distance = np.linalg.norm(offset, axis=-1)
    unit = offset / np.maximum(distance[:, :, None], 0.1)
    visible = np.isfinite(ball).all(axis=1)
    count_columns, speed_columns = [], []
    for side in (0, 1):
        group = distance[:, half.sides == side]
        observed = visible & np.isfinite(group).any(axis=1)
        for radius in (2, 5, 10):
            count_columns.append(np.where(observed, (group <= radius).sum(axis=1), np.nan))
    # 分别观察接触前后的运动，同一轨迹计算速度，连续缺球处不计算相对球速。
    missing = np.r_[0, np.cumsum(~np.isfinite(half.ball).all(axis=1))]
    for start, stop in ((np.maximum(clock - 6, 0), clock),
                        (clock, np.minimum(clock + 6, len(half.ball) - 1))):
        safe_stop = np.maximum(stop, start + 1)
        safe_start = np.minimum(start, len(half.ball) - 2)
        safe_stop = np.minimum(safe_stop, len(half.ball) - 1)
        player_velocity = velocity_between(half.players, safe_start, safe_stop, fps, 12)
        ball_velocity = velocity_between(half.ball, safe_start, safe_stop, fps, 45)
        ball_velocity[(missing[stop + 1] != missing[start]) | (stop == start)] = np.nan
        player_velocity[stop == start] = np.nan
        radial = np.sum(player_velocity * unit, axis=-1)
        relative = player_velocity - ball_velocity[:, None, :]
        closing = np.sum(relative * unit, axis=-1)
        relative_speed = np.linalg.norm(relative, axis=-1)
        for side in (0, 1):
            slots = np.flatnonzero(half.sides == side)
            ds = distance[:, slots]
            order = np.argsort(np.where(np.isfinite(ds), ds, np.inf), axis=1)
            nearest = slots[order[:, 0]]
            row = np.arange(len(clock))
            speed_columns.extend((radial[row, nearest], closing[row, nearest], relative_speed[row, nearest]))
            local = np.isfinite(radial[:, slots]) & (ds <= 10)
            fastest = np.where(local, radial[:, slots], -np.inf).max(axis=1)
            speed_columns.append(np.where(np.isfinite(fastest), fastest, np.nan))
    return np.column_stack(count_columns).astype(np.float32), np.column_stack(speed_columns).astype(np.float32)
