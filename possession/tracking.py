from dataclasses import dataclass

import numpy as np
from kloppy import metrica
from lxml import etree

from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter import load_smart_tagging
from starter.baseline import team_names


@dataclass
class Half:
    players: np.ndarray
    ball: np.ndarray
    sides: np.ndarray
    pitch: np.ndarray


def read_target(data_dir):
    meta, raw = require(['atd_meta', 'atd_raw'], data_dir=data_dir)
    tracking = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    names = team_names(meta)
    if tuple(team.name for team in tracking.metadata.teams) != names:
        raise ValueError('球队顺序与 EPTS 元数据不一致')
    roster = [(player.player_id, side) for side, team in enumerate(tracking.metadata.teams)
              for player in team.players]
    slots = {name: i for i, (name, _) in enumerate(roster)}
    sides = np.array([side for _, side in roster])
    n = source_frames()
    if len(slots) != len(sides) or len(tracking.records) != n:
        raise ValueError('球员槽位重复或 ATD 帧数错误')
    players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    for i, frame in enumerate(tracking.records):
        if int(frame.frame_id) != i + 1:
            raise ValueError('战术跟踪帧号不连续')
        point = frame.ball_coordinates
        if point is not None:
            ball[i] = point.x, point.y
        for player, data in frame.players_data.items():
            point = data.coordinates
            if point is not None:
                players[i, slots[player.player_id]] = point.x, point.y
    del tracking
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    players *= pitch
    ball *= pitch
    halves, masks, starts = [], [], []
    blocks = half_video_blocks(GAME)
    kickoffs = load_smart_tagging.KICKOFF_VIDEO_FRAMES
    for (_, start, end), period in zip(blocks, ('1st Half', '2nd Half'), strict=True):
        kickoff = kickoffs[period]
        if not start <= kickoff <= end:
            raise ValueError('官方公布的开球时间超出该半场')
        masks.append(np.isfinite(ball[start:end + 1]).all(axis=1))
        starts.append(kickoff)
        halves.append(Half(players[kickoff:end + 1], ball[kickoff:end + 1], sides, pitch))
    return halves, masks, starts, names, load_smart_tagging.load(data_dir=data_dir)


def read_idsse(directory, match_id):
    meta_path, = directory.glob(f'DFL_02_01_*{match_id}.xml')
    pos_path, = directory.glob(f'DFL_04_03_*{match_id}.xml')
    meta = etree.parse(str(meta_path))
    env, = meta.xpath('//Environment')
    pitch = np.array([env.attrib['PitchX'], env.attrib['PitchY']], dtype=np.float32)
    team_ids = [meta.xpath(f'//Team[@Role="{role}"]/@TeamId')[0] for role in ('home', 'guest')]
    roster = [(player, side) for side, team in enumerate(team_ids)
              for player in meta.xpath('//Team[@TeamId=$team]//Player/@PersonId', team=team)]
    slots = {player: i for i, (player, _) in enumerate(roster)}
    sides = np.asarray([side for _, side in roster])
    periods = {'firstHalf': [], 'secondHalf': []}
    for _, group in etree.iterparse(str(pos_path), tag='FrameSet', huge_tree=True):
        team = group.attrib['TeamId']
        if team in team_ids or team == 'BALL':
            def column(name, dtype):
                return np.asarray(group.xpath(f'Frame/@{name}', smart_strings=False), dtype=dtype)

            ids = column('N', np.int32)
            if len(ids) == 0 or (np.diff(ids) <= 0).any():
                raise ValueError(f'{match_id} 跟踪帧号无效')
            xy = np.column_stack((column('X', np.float32), column('Y', np.float32))) + pitch / 2
            if len(xy) != len(ids):
                raise ValueError(f'{match_id} 缺少坐标属性')
            slot, labels = None, None
            if team == 'BALL':
                status = column('BallStatus', np.int8)
                owner = column('BallPossession', np.int8)
                if (len(status) != len(ids) or len(owner) != len(ids)
                        or not np.isin(status, [0, 1]).all() or not np.isin(owner, [0, 1, 2]).all()):
                    raise ValueError(f'{match_id} 控球标注缺失或编码未知')
                labels = np.select([status == 0, owner == 1, owner == 2], [2, 0, 1], default=3).astype(np.int8)
            else:
                slot = slots[group.attrib['PersonId']]
                if team_ids[sides[slot]] != team:
                    raise ValueError(f'{match_id} 球员所属球队不一致')
            periods[group.attrib['GameSection']].append((ids, xy, slot, labels))
        group.clear()
        while group.getprevious() is not None:
            del group.getparent()[0]

    halves, truths = [], []
    for groups in periods.values():
        ball_ids = np.concatenate([ids for ids, _, slot, _ in groups if slot is None])
        if len(np.unique(ball_ids)) != len(ball_ids):
            raise ValueError(f'{match_id} 足球帧号重复')
        start, end = int(ball_ids.min()), int(ball_ids.max())
        n = end - start + 1
        players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        truth = np.full(n, 3, dtype=np.int8)
        for ids, xy, slot, labels in groups:
            within = (ids >= start) & (ids <= end)
            indices = ids[within] - start
            if slot is None:
                ball[indices] = xy[within]
                truth[indices] = labels[within]
            else:
                if np.isfinite(players[indices, slot, 0]).any():
                    raise ValueError(f'{match_id} 球员坐标重复')
                players[indices, slot] = xy[within]
        halves.append(Half(players, ball, sides, pitch))
        truths.append(truth)
    return halves, truths
