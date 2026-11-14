"""Measure how the competition's tracking breaks, so practice matches can be damaged the same way.

In short
--------
The model learns on public matches whose clean tracking test_on_other_matches.damage breaks on
purpose, to look like the competition's tracking (the ATD). DATA.md gives only the shares of missing
ball and players. How long the gaps last, where they happen and how noisy the positions are, it
does not. This file prints the same whole-match numbers for the ATD and for the damaged public
matches, and the damage settings in test_on_other_matches.py were set so that the two agree.

It was used once, to set those values. The model does not need it to run.

    python starter/measure_tracking.py            # the provided match: one table
    python starter/measure_tracking.py --other    # the same table for the IDSSE matches

What it reads and prints
------------------------
Only the tracking, and only the frames inside the two halves. No ground truth, no tags, nothing
about who has the ball. Every number is a share, a rate or a percentile over the whole match, so
none is tied to particular frames (RULES.md) and none could rebuild the tracking.

The stoppage numbers use the players instead of a label: a frame is calm when the players' mean
speed over two seconds is below possession_v1's walking pace, which is how that file sees a dead
ball, and active when it is above its playing pace.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (ROOT, HERE):
    sys.path.insert(0, folder)
from baseline import AWAY, HOME, team_of_each_track  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from possession_model import read_full_tracking  # noqa: E402
from possession_v1 import (DEAD_WINDOW, FPS, PLAYING_SPEED, WALKING_SPEED,  # noqa: E402
                           rolling_mean)

# Params
CROWD_M = 3.0                       # players this close to the ball crowd it, as in damage()
SPEED_LAG = 4                       # frames back over which the ball's speed before a gap is taken
BALL_SPEEDS = (3.0, 10.0, 20.0)     # m/s, edges of the ball-speed groups
JUMP_M = 2.0                        # a one-frame move of 50 m/s, faster than any kick
FAR_M = 15.0
EDGE_M = 1.0                        # how far past a line counts as outside the pitch
BLIP_FRAMES = 3
BOX_DEPTH_M, BOX_WIDTH_M = 16.5, 40.32
SPEED_CHECK_LAG = 5                 # frames either side when working a speed out from positions
SLOW_BEND = 6                       # frames either side for the bend that slow errors show up in
# A ball that moves but does not bend over SLOW_BEND frames for a second or more is more likely to
# have been drawn as a straight line across a gap by the tracker than to have really moved so.
STRAIGHT_M, STRAIGHT_SPEED, STRAIGHT_FRAMES = 0.02, 1.0, FPS
GAP_HISTOGRAM_S = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 60.0)
SQUAD = 22
GAP_LENGTHS_S = (1.0, 3.0, 10.0)    # edges of the gap-length groups for how far the ball moved
SAMPLE_EVERY = 5                    # player numbers use every fifth frame; percentiles barely move
CALM, BETWEEN, ACTIVE = 0, 1, 2


def runs(mask):
    """(starts, stops) of each run of True in a 1-D mask, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def pct(values, q):
    """The q-th percentile, ignoring NaN. NaN if there is nothing to take it of."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[~np.isnan(values)]
    return float(np.percentile(values, q)) if len(values) else np.nan


def half_parts(ball, xy, speed, team, pitch):
    """Per-frame and per-gap arrays for one half, which summary() turns into numbers."""
    length, width = pitch
    n_frames = len(ball)
    team = np.broadcast_to(team, xy.shape[:2])
    seen = ~np.isnan(ball[:, 0])
    tracked = ~np.isnan(xy[..., 0])

    to_ball = np.hypot(xy[..., 0] - ball[:, 0, np.newaxis], xy[..., 1] - ball[:, 1, np.newaxis])
    crowd = np.sum(to_ball < CROWD_M, axis=1)
    reach = np.where(np.isnan(to_ball), np.inf, to_ball).min(axis=1)
    nearest = np.where(np.isfinite(reach), reach, np.nan)

    timed = tracked & ~np.isnan(speed)
    n_timed = timed.sum(axis=1)
    crowd_speed = np.full(n_frames, np.nan)
    has_speed = n_timed > 0
    crowd_speed[has_speed] = np.where(timed, speed, 0.0).sum(axis=1)[has_speed] / n_timed[has_speed]
    calm_speed = rolling_mean(crowd_speed, DEAD_WINDOW)
    state = np.full(n_frames, -1, dtype=np.int8)
    state[calm_speed < WALKING_SPEED] = CALM
    state[(calm_speed >= WALKING_SPEED) & (calm_speed <= PLAYING_SPEED)] = BETWEEN
    state[calm_speed > PLAYING_SPEED] = ACTIVE

    before = np.full(n_frames, np.nan)
    after = np.full(n_frames, np.nan)
    moved = ball[SPEED_LAG:] - ball[:-SPEED_LAG]
    before[SPEED_LAG:] = np.hypot(moved[:, 0], moved[:, 1]) * FPS / float(SPEED_LAG)
    after[:-SPEED_LAG] = before[SPEED_LAG:]
    x, y = ball[:, 0], ball[:, 1]
    in_box = (((x < BOX_DEPTH_M) | (x > length - BOX_DEPTH_M))
              & (np.abs(y - width / 2.0) < BOX_WIDTH_M / 2.0))
    across = np.digitize(y, (width / 3.0, 2.0 * width / 3.0))
    along = np.digitize(x, (length / 3.0, 2.0 * length / 3.0))
    outside = seen & ((x < -EDGE_M) | (x > length + EDGE_M) | (y < -EDGE_M) | (y > width + EDGE_M))

    # A gap starts after frame t when t is the last frame the ball is seen on.
    at_risk = np.zeros(n_frames, dtype=bool)
    at_risk[:-1] = seen[:-1]
    starts_gap = np.zeros(n_frames, dtype=bool)
    starts_gap[:-1] = seen[:-1] & ~seen[1:]

    gap_starts, gap_stops = runs(~seen)
    inner = (gap_starts > 0) & (gap_stops < n_frames)
    gap_move = np.full(len(gap_starts), np.nan)
    across_gap = ball[gap_stops[inner]] - ball[gap_starts[inner] - 1]
    gap_move[inner] = np.hypot(across_gap[:, 0], across_gap[:, 1])
    after_seen = gap_starts > 0
    gap_crowd = np.full(len(gap_starts), -1)
    gap_crowd[after_seen] = crowd[gap_starts[after_seen] - 1]
    gap_state = np.full(len(gap_starts), -1, dtype=np.int8)
    gap_state[after_seen] = state[gap_starts[after_seen] - 1]
    ends = gap_stops[gap_stops < n_frames]  # the first frame the ball is seen again
    seen_starts, seen_stops = runs(seen)

    bend = ball[2:] - 2.0 * ball[1:-1] + ball[:-2]
    slow = SLOW_BEND
    slow_bend = ball[2 * slow:] - 2.0 * ball[slow:-slow] + ball[:-2 * slow]
    step = np.diff(ball, axis=0)
    bent = np.full(n_frames, np.nan)
    bent[slow:n_frames - slow] = np.hypot(slow_bend[:, 0], slow_bend[:, 1])
    pace = np.full(n_frames, np.nan)
    across_two = ball[2:] - ball[:-2]
    pace[1:-1] = np.hypot(across_two[:, 0], across_two[:, 1]) * FPS / 2.0
    straight_starts, straight_stops = runs((bent < STRAIGHT_M) & (pace > STRAIGHT_SPEED))
    straight_frames = straight_stops - straight_starts

    stretches = []
    for slot in range(xy.shape[1]):
        first, last = runs(tracked[:, slot])
        stretches.append(last - first)

    sample = slice(0, None, SAMPLE_EVERY)
    lag = SPEED_CHECK_LAG
    player_bend = (xy[2:] - 2.0 * xy[1:-1] + xy[:-2])[sample]
    player_slow_bend = (xy[2 * slow:] - 2.0 * xy[slow:-slow] + xy[:-2 * slow])[sample]
    player_step = (xy[1:] - xy[:-1])[sample]
    walked = (xy[2 * lag:] - xy[:-2 * lag])[sample]
    from_positions = np.hypot(walked[..., 0], walked[..., 1]) * FPS / (2.0 * lag)

    n_tracked = tracked.sum(axis=1)
    over_starts, over_stops = runs(n_tracked > SQUAD)
    return {"frames": n_frames, "seen": seen, "crowd": crowd, "nearest": nearest, "state": state,
            "before": before, "after": after, "in_box": in_box, "across": across, "along": along,
            "outside": outside, "at_risk": at_risk, "starts_gap": starts_gap,
            "gap_frames": gap_stops - gap_starts, "gap_move": gap_move, "gap_crowd": gap_crowd,
            "gap_state": gap_state, "end_crowd": crowd[ends], "end_after": after[ends],
            "end_state": state[ends], "seen_frames": seen_stops - seen_starts,
            "straight_frames": straight_frames[straight_frames >= STRAIGHT_FRAMES],
            "over_frames": over_stops - over_starts,
            "ball_jitter": np.hypot(bend[:, 0], bend[:, 1]),
            "ball_slow_bend": np.hypot(slow_bend[:, 0], slow_bend[:, 1]),
            "ball_step": np.hypot(step[:, 0], step[:, 1]),
            "n_tracked": n_tracked, "n_home": (tracked & (team == HOME)).sum(axis=1),
            "n_away": (tracked & (team == AWAY)).sum(axis=1),
            "stretch_frames": np.concatenate(stretches), "crowd_speed": crowd_speed,
            "player_jitter": np.hypot(player_bend[..., 0], player_bend[..., 1]).ravel(),
            "player_slow_bend": np.hypot(player_slow_bend[..., 0],
                                         player_slow_bend[..., 1]).ravel(),
            "player_step": np.hypot(player_step[..., 0], player_step[..., 1]).ravel(),
            "speed_gap": np.abs(from_positions - speed[lag:-lag][sample]).ravel(),
            "speeds": speed[sample][timed[sample]]}


def speed_groups():
    """[(label, low, high)] for the ball-speed groups, in m/s."""
    edges = (0.0,) + BALL_SPEEDS + (np.inf,)
    groups = []
    for low, high in zip(edges[:-1], edges[1:]):
        if low == 0.0:
            label = "ball under {:.0f} m/s".format(high)
        elif np.isinf(high):
            label = "ball over {:.0f} m/s".format(low)
        else:
            label = "ball {:.0f}-{:.0f} m/s".format(low, high)
        groups.append((label, low, high))
    return groups


def summary(parts):
    """[(label, value, unit)] over every half in parts, in print order."""
    joined = {key: np.concatenate([np.atleast_1d(part[key]) for part in parts]) for key in parts[0]}
    rows = []

    def add(label, value, unit):
        rows.append((label, value, unit))

    def section(title):
        rows.append((title, None, "section"))

    minutes = joined["frames"].sum() / float(FPS) / 60.0
    seen = joined["seen"]
    gap_frames = joined["gap_frames"]
    gap_s = gap_frames / float(FPS)
    add("graded minutes", minutes, "min")

    section("Ball gaps")
    add("ball missing", np.mean(~seen), "share")
    add("gaps per minute", len(gap_s) / minutes, "/min")
    for name, q in (("10th percentile", 10), ("25th percentile", 25), ("median", 50),
                    ("75th percentile", 75), ("90th percentile", 90), ("99th percentile", 99)):
        add("gap length, " + name, pct(gap_s, q), "s")
    for edge in (2.0, 10.0, 60.0):
        add("missing frames in gaps over {:.0f} s".format(edge),
            gap_frames[gap_s > edge].sum() / float(max(gap_frames.sum(), 1)), "share")
    add("gaps of a single frame", np.mean(gap_frames == 1) if len(gap_frames) else np.nan, "share")
    seen_s = joined["seen_frames"] / float(FPS)
    add("visible stretch, median", pct(seen_s, 50), "s")
    add("visible stretch, 90th percentile", pct(seen_s, 90), "s")
    add("visible stretches of {} frames or fewer".format(BLIP_FRAMES),
        np.mean(joined["seen_frames"] <= BLIP_FRAMES), "share")

    section("Gaps by length (share of gaps)")
    edges = (0.0,) + GAP_HISTOGRAM_S + (np.inf,)
    for low, high in zip(edges[:-1], edges[1:]):
        if low == 0.0:
            label = "up to {:.0f} s".format(high)
        elif np.isinf(high):
            label = "over {:.0f} s".format(low)
        else:
            label = "{:.0f}-{:.0f} s".format(low, high)
        add(label, np.mean((gap_s > low) & (gap_s <= high)) if len(gap_s) else np.nan, "share")

    at_risk, starts = joined["at_risk"], joined["starts_gap"]
    state = joined["state"]

    def hazard(label, mask, base):
        """How often a gap starts where mask holds, against the rate over base, and how much of
        base mask covers."""
        n_base = float(max((at_risk & base).sum(), 1))
        average = starts[at_risk & base].sum() / n_base
        risk = at_risk & base & mask
        n_risk = risk.sum()
        ratio = starts[risk].sum() / float(n_risk) / average if n_risk and average > 0 else np.nan
        add(label, (ratio, n_risk / n_base), "ratio")

    section("Where gaps start: rate against the average (share of visible frames)")
    everywhere = np.ones(len(state), dtype=bool)
    hazard("players calm", state == CALM, everywhere)
    hazard("players in between", state == BETWEEN, everywhere)
    hazard("players active", state == ACTIVE, everywhere)

    # Stoppages lose the ball most and keep players away from it, so the rest is in active play.
    section("The same with the players active, against their own average")
    active = state == ACTIVE
    crowd = joined["crowd"]
    hazard("no player within 3 m", crowd == 0, active)
    hazard("1 or 2 players within 3 m", (crowd >= 1) & (crowd <= 2), active)
    hazard("3 or more players within 3 m", crowd >= 3, active)
    before = joined["before"]
    for label, low, high in speed_groups():
        hazard(label, (before >= low) & (before < high), active)
    hazard("ball in a penalty box", joined["in_box"], active)
    hazard("ball outside the boxes", ~joined["in_box"], active)
    across, along = joined["across"], joined["along"]
    hazard("ball in the third of the width nearest y = 0", across == 0, active)
    hazard("ball in the middle third of the width", across == 1, active)
    hazard("ball in the third of the width furthest from y = 0", across == 2, active)
    hazard("ball in the third of the length nearest x = 0", along == 0, active)
    hazard("ball in the middle third of the length", along == 1, active)
    hazard("ball in the third of the length furthest from x = 0", along == 2, active)
    hazard("nearest player over {:.0f} m away".format(FAR_M), joined["nearest"] > FAR_M, active)

    section("Gap length, by the situation it starts in")
    gap_crowd, gap_state = joined["gap_crowd"], joined["gap_state"]
    for name, code in (("calm", CALM), ("active", ACTIVE)):
        add("players {}, median".format(name), pct(gap_s[gap_state == code], 50), "s")
        add("players {}, 90th percentile".format(name), pct(gap_s[gap_state == code], 90), "s")
    in_play = gap_state == ACTIVE
    add("active, no player within 3 m, median", pct(gap_s[in_play & (gap_crowd == 0)], 50), "s")
    add("active, 1 or 2 players within 3 m, median",
        pct(gap_s[in_play & (gap_crowd >= 1) & (gap_crowd <= 2)], 50), "s")
    add("active, 3 or more players within 3 m, median",
        pct(gap_s[in_play & (gap_crowd >= 3)], 50), "s")

    section("Where the ball is found again, players active: against all active visible frames "
            "(share of finds)")
    seen_active = seen & active
    n_seen_active = float(max(seen_active.sum(), 1))
    end_active = joined["end_state"] == ACTIVE
    n_ends = float(max(end_active.sum(), 1))

    def found(label, at_end, at_frame):
        share_end = (end_active & at_end).sum() / n_ends
        share_all = (seen_active & at_frame).sum() / n_seen_active
        add(label, (share_end / share_all if share_all > 0 else np.nan, share_end), "ratio")

    end_crowd, end_after, after = joined["end_crowd"], joined["end_after"], joined["after"]
    found("no player within 3 m", end_crowd == 0, crowd == 0)
    found("1 or 2 players within 3 m", (end_crowd >= 1) & (end_crowd <= 2),
          (crowd >= 1) & (crowd <= 2))
    found("3 or more players within 3 m", end_crowd >= 3, crowd >= 3)
    for label, low, high in speed_groups():
        found(label + " just after", (end_after >= low) & (end_after < high),
              (after >= low) & (after < high))

    section("How far the ball moves across a gap, median")
    move = joined["gap_move"]
    edges = (0.0,) + GAP_LENGTHS_S + (np.inf,)
    for low, high in zip(edges[:-1], edges[1:]):
        if low == 0.0:
            label = "gaps up to {:.0f} s".format(high)
        elif np.isinf(high):
            label = "gaps over {:.0f} s".format(low)
        else:
            label = "gaps of {:.0f}-{:.0f} s".format(low, high)
        add(label, pct(move[(gap_s > low) & (gap_s <= high)], 50), "m")

    section("Stoppages, as the players show them")
    known = state >= 0
    add("frames with the players calm", np.mean(state[known] == CALM), "share")
    add("frames with the players active", np.mean(state[known] == ACTIVE), "share")
    for label, code in (("ball missing when calm", CALM), ("ball missing in between", BETWEEN),
                        ("ball missing when active", ACTIVE)):
        here = state == code
        add(label, np.mean(~seen[here]) if here.any() else np.nan, "share")

    section("The ball while it is visible")
    add("jitter (frame-to-frame bend), median", pct(joined["ball_jitter"], 50), "m")
    add("jitter, 90th percentile", pct(joined["ball_jitter"], 90), "m")
    add("bend over {} frames, median".format(SLOW_BEND), pct(joined["ball_slow_bend"], 50), "m")
    step = joined["ball_step"]
    step = step[~np.isnan(step)]
    add("exactly still from one frame to the next",
        np.mean(step == 0.0) if len(step) else np.nan, "share")
    add("one-frame jumps over {:.0f} m".format(JUMP_M),
        np.mean(step > JUMP_M) if len(step) else np.nan, "share")
    nearest = joined["nearest"]
    visible = nearest[seen & ~np.isnan(nearest)]
    add("nearest player over {:.0f} m away".format(FAR_M),
        np.mean(visible > FAR_M) if len(visible) else np.nan, "share")
    add("outside the pitch by over {:.0f} m".format(EDGE_M),
        joined["outside"].sum() / float(max(seen.sum(), 1)), "share")
    for name, q in (("25th percentile", 25), ("median", 50), ("75th percentile", 75)):
        add("nearest player, " + name, pct(visible, q), "m")

    section("Straight-line stretches of the ball: moving, bending under {:.0f} cm over {} frames, "
            "for {:.0f} s or more".format(STRAIGHT_M * 100, SLOW_BEND, STRAIGHT_FRAMES / FPS))
    straight_s = joined["straight_frames"] / float(FPS)
    add("stretches per minute", len(straight_s) / minutes, "/min")
    add("length, median", pct(straight_s, 50), "s")
    add("length, 90th percentile", pct(straight_s, 90), "s")
    add("share of visible frames inside them",
        joined["straight_frames"].sum() / float(max(seen.sum(), 1)), "share")

    section("Players")
    tracked = joined["n_tracked"]
    add("tracked per frame, mean", np.mean(tracked), "count")
    for name, q in (("10th percentile", 10), ("median", 50), ("90th percentile", 90)):
        add("tracked per frame, " + name, pct(tracked, q), "count")
    add("home players tracked, mean", np.mean(joined["n_home"]), "count")
    add("away players tracked, mean", np.mean(joined["n_away"]), "count")
    add("frames with more than 22 tracked", np.mean(tracked > SQUAD), "share")
    add("more than 22 tracked, median spell", pct(joined["over_frames"] / float(FPS), 50), "s")
    short = SQUAD - tracked
    for label, low, high in (("all 22 or more tracked", -SQUAD, 0), ("1 missing", 1, 1),
                             ("2 missing", 2, 2), ("3 or 4 missing", 3, 4),
                             ("5 to 8 missing", 5, 8), ("9 or more missing", 9, SQUAD)):
        add("frames with " + label, np.mean((short >= low) & (short <= high)), "share")
    for label, mask in (("length third near x = 0", along == 0),
                        ("middle length third", along == 1),
                        ("length third far from x = 0", along == 2),
                        ("width third near y = 0", across == 0),
                        ("width third far from y = 0", across == 2),
                        ("penalty box", joined["in_box"])):
        here = seen & mask
        add("tracked, ball in the " + label,
            np.mean(tracked[here]) if here.any() else np.nan, "count")
    stretch_s = joined["stretch_frames"] / float(FPS)
    for name, q in (("10th percentile", 10), ("median", 50), ("90th percentile", 90)):
        add("unbroken track, " + name, pct(stretch_s, q), "s")
    add("jitter (frame-to-frame bend), median", pct(joined["player_jitter"], 50), "m")
    add("jitter, 90th percentile", pct(joined["player_jitter"], 90), "m")
    add("bend over {} frames, median".format(SLOW_BEND), pct(joined["player_slow_bend"], 50), "m")
    player_step = joined["player_step"]
    player_step = player_step[~np.isnan(player_step)]
    add("exactly still from one frame to the next",
        np.mean(player_step == 0.0) if len(player_step) else np.nan, "share")
    add("speed, median", pct(joined["speeds"], 50), "m/s")
    add("speed, 99th percentile", pct(joined["speeds"], 99), "m/s")
    for name, q in (("10th percentile", 10), ("median", 50), ("90th percentile", 90)):
        add("mean speed of everyone, " + name, pct(joined["crowd_speed"], q), "m/s")
    add("speed column against positions, median gap", pct(joined["speed_gap"], 50), "m/s")
    return rows


def show(value, unit):
    """One value as printed."""
    if unit == "ratio":
        ratio, share = value
        return "{} ({:.0%})".format("-" if np.isnan(ratio) else "x{:.2f}".format(ratio), share)
    if value is None or np.isnan(value):
        return "-"
    formats = {"share": "{:.1%}", "s": "{:.2f} s", "m": "{:.2f} m", "m/s": "{:.2f} m/s",
               "/min": "{:.2f} /min", "count": "{:.1f}", "min": "{:.1f} min"}
    return formats[unit].format(value)


def print_table(titles, tables):
    """Tables of the same rows side by side, one column each."""
    width = 54
    print("{:<{w}}{}".format("", "".join("{:>16}".format(title) for title in titles), w=width))
    for rows in zip(*tables):
        label, _, unit = rows[0]
        if unit == "section":
            print("\n" + label)
            continue
        print("  {:<{w}}{}".format(label, "".join("{:>16}".format(show(value, kind))
                                                   for _, value, kind in rows), w=width - 2))


def read_provided(game=GAME, data_dir=None):
    """half_parts for each half of the provided match, from its ATD."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)
    team_of_track = team_of_each_track(meta_path)
    print("reading {} ...".format(os.path.basename(raw_path)))
    ball, xy, speed = read_full_tracking(raw_path, team_of_track, source_frames("tactical", game),
                                         pitch, blocks)
    return [half_parts(ball[first:last + 1], xy[first:last + 1], speed[first:last + 1],
                       team_of_track, pitch) for _, first, last in blocks]


def read_other(names):
    """(damaged, clean) half_parts over the IDSSE matches named, damaged as test_on_other_matches
    scores them, and how far the calm and active frames there really are stoppages."""
    import test_on_other_matches as other
    import possession_metrics as pm
    damaged, clean, dead_when = [], [], {CALM: [], ACTIVE: [], "all": []}
    for stem in other.MATCHES:
        if names and stem[-6:] not in names:
            continue
        print("measuring {} ...".format(stem[-6:]), flush=True)
        match = other.load_match(stem, other.DATA_DIR)
        broken = other.damage(match, other.SEED + other.MATCHES.index(stem))
        for _, first, last in match["blocks"]:
            stop = last + 1
            for source, out in ((broken, damaged), (match, clean)):
                out.append(half_parts(source["ball"][first:stop], source["xy"][first:stop],
                                      source["speed"][first:stop], source["team_of"][first:stop],
                                      match["pitch"]))
            dead = match["truth"][first:stop] == pm.D
            state = damaged[-1]["state"]
            dead_when[CALM].append(dead[state == CALM])
            dead_when[ACTIVE].append(dead[state == ACTIVE])
            dead_when["all"].append(dead)
    shares = {key: float(np.mean(np.concatenate(value))) for key, value in dead_when.items()}
    return damaged, clean, shares


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--other", action="store_true",
                        help="measure the IDSSE matches instead, damaged and clean")
    parser.add_argument("--matches", nargs="*", help="IDSSE codes for --other, such as J03WN1")
    args = parser.parse_args()
    if args.other:
        damaged, clean, shares = read_other(args.matches)
        print()
        print_table(("damaged", "clean"), [summary(damaged), summary(clean)])
        print("\nOn these matches the stoppages are {:.1%} of frames, {:.1%} of the calm frames "
              "and {:.1%} of the active ones.".format(shares["all"], shares[CALM], shares[ACTIVE]))
    else:
        parts = read_provided()
        print()
        print_table(("your match",), [summary(parts)])
        print("\nOnly whole-match numbers are printed above, nothing about single moments.")
