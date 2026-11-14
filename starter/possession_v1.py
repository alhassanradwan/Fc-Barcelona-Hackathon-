"""Possession v1: football rules give every frame points, then Viterbi picks the labels.

In short
--------
The task is to say, for every video frame, who has the ball: H (home team), A (away team), D (dead
ball, play stopped), or X outside the two halves. This file does it with no machine learning, in
three steps:

  1. Measure. For every frame: how close each team is to the ball, how fast the ball and the
     players move, and whether the ball is off the pitch.
  2. Give points. Each frame gives points to H, A and D, by the table below. A frame where nobody
     clearly has the ball gives no team any points.
  3. Pick the labels. The Viterbi algorithm (best_labels) finds the run of labels with the most
     points in total, where every change of label costs points.

There is no training and no randomness, so the same data always gives the same file. It is the next
step up from baseline.py, and possession_v2.py and possession_model.py build on its code.

    pip install -r requirements.txt          # numpy and pandas, nothing else
    python starter/possession_v1.py          # writes evaluation/results/possession_v1.csv
    python evaluation/validate_submission.py evaluation/results/possession_v1.csv
    python evaluation/run_evaluation.py --no-detail

Expect a few minutes: it reads 199,319 lines of tracking.

Who gets the points
-------------------
    ball close to a player and not flying past them     that player's team
    both teams equally close                            nobody: a contest stays with the last team
    ball flying, or not seen at all                     nobody
    ball off the pitch, or still while players walk     D
    no sign of a dead ball                              H and A equally, which only argues against D
    Smart Tagging set piece                             D before the restart, then the tagged team
    Smart Tagging phase                                 a little, to the team the phase implies
    before a half's "Kick Off (Start)" tag              D

Why every change of label costs points
--------------------------------------
- One odd frame cannot pay for a change, so the labels do not flicker. (baseline.py needs a
  smoothing window for that.)
- A frame with no evidence earns nothing, so the current label simply carries on. That is how the
  gaps where the ball is missing from the tracking get filled.
- A quick deflection earns too little to pay for two changes (there and back again). That matches
  CHALLENGE.md: a touch without control is not a change of possession.
- Viterbi sees the whole half at once, so a change lands where the evidence flips, not wherever a
  smoothing window happens to put it.

Why these values
----------------
Every value under Params is a starting guess from football common sense. None was chosen by looking
at a ground truth, and none should be: the public five minutes are not a tuning set (RULES.md).
possession_v2.py tunes its values on other matches instead. No frame number of this match is
written anywhere in this file either, because RULES.md rules out constants tied to specific frames.

Known limits
------------
- A goalkeeper holding the ball while everyone walks looks like a dead ball.
- A ball being carried to a restart looks alive until it stops moving.
- A long stretch without the ball keeps the last team, right or wrong.
- The Team tag on a defending phase is read as the defending team, following the definitions in
  OVERVIEW.md. That reading has not been checked against the data.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, LABELS, MISSING,  # noqa: E402
                      RESULTS_DIR, team_names, team_of_each_track, write_submission)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import KICKOFF_TYPE, load as load_smart_tagging  # noqa: E402

# Params
OUT_NAME = "possession_v1.csv"
FPS = GAMES[GAME]["fps"]
DEAD = 2  # H and A are baseline's HOME and AWAY, so LABELS turns a state straight into its letter

# Control. Full points with the ball this close to a player, none at the second distance, in metres.
CONTROL_NEAR, CONTROL_FAR = 1.0, 4.0
# How much faster than that player the ball may be moving, in m/s. Below the first it can be at
# their feet; above the second it is flying past them.
CARRIED_EXTRA, FLYING_EXTRA = 4.0, 10.0
# Frames either side used to measure ball speed. Short, so a kick shows up at once.
SPEED_LAG = 2

# Dead ball. Mean player speed, in m/s: full points at walking pace, none at the second value, which
# is a little above the roughly 2 m/s that 10-11 km over 90 minutes works out at.
WALKING_SPEED, PLAYING_SPEED = 1.2, 2.2
# The ball counts as still if it moves slower than this, in m/s, over +/- STILL_LAG frames. Long, so
# tracking jitter on a stationary ball is not read as movement.
STILL_SPEED, STILL_LAG = 1.0, 12
# Both of the above are averaged over +/- this many frames, because a stoppage is a state and a
# single quiet frame says nothing.
DEAD_WINDOW = FPS
# Metres beyond the line, and for how long, before the ball counts as off the pitch. A high ball can
# project past the line for a moment, so one frame is not enough.
OFF_PITCH_MARGIN, OFF_PITCH_WINDOW = 1.0, 12

# Smart Tagging. All of this is evidence and never an answer -- see DATA.md.
SET_PIECES_CODE = "set pieces"
SET_PIECE_POINTS = 0.5
# A set piece is tagged as a window around the restart, opening a few seconds early, so the restart
# is taken to be this far through it. The tracking can move the change from there.
RESTART_POSITION = 0.6
AFTER_RESTART = FPS
PHASE_POINTS = 0.1
BEFORE_KICKOFF_POINTS = 5.0
IN_POSSESSION_CODES = ("build up", "progression", "creating chances", "attacking transition",
                       "long ball", "ball in final third", "ball in the box", "players in the box")
OUT_OF_POSSESSION_CODES = ("defending in attacking third", "defending in middle third",
                           "defending in defensive third", "defensive transition")

# What a change of label costs. One frame of clear evidence is worth about one point, so 25 points
# is about a second. A brief spell by the other team pays twice, to change and to change back, so
# SWITCH_COST asks for about a second of clear control. A stoppage also pays twice, to start and to
# end, so STOP_COST and RESTART_COST ask for about two seconds of clear evidence.
SWITCH_COST = 12.5
STOP_COST = 25.0
RESTART_COST = 25.0
COSTS = ((0.0, SWITCH_COST, STOP_COST),
         (SWITCH_COST, 0.0, STOP_COST),
         (RESTART_COST, RESTART_COST, 0.0))
TIE = 1e-6  # scores this close are equal; they differ only by rounding

# A median player speed outside this range means the speed column is not what DATA.md says it is.
PLAUSIBLE_SPEED = (0.5, 5.0)


def read_tracking(raw_path, team_of_track, n_frames, pitch):
    """(ball, nearest, nearest_speed, crowd_speed), each with one row per video frame.

    ball is (n, 2) in metres. nearest and nearest_speed are (n, 2) with one column per team: the
    distance from the ball to that team's closest player, and that player's speed. crowd_speed is
    (n,), the mean speed of everyone tracked. All of them are NaN where there is nothing to report.
    """
    length, width = pitch
    teams = team_of_track.tolist()
    ball = np.full((n_frames, 2), np.nan)
    nearest = np.full((n_frames, 2), np.nan)
    nearest_speed = np.full((n_frames, 2), np.nan)
    crowd_speed = np.full(n_frames, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0

            ball_seen = parts[2][0] != MISSING
            if ball_seen:
                fields = parts[2].split(",")
                ball_x, ball_y = float(fields[0]) * length, float(fields[1]) * width
                ball[frame] = ball_x, ball_y

            closest = [math.inf, math.inf]
            closest_speed = [math.nan, math.nan]
            total_speed, n_speeds = 0.0, 0
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                fields = group.split(",")
                speed = float(fields[2])
                if not math.isnan(speed):
                    total_speed += speed
                    n_speeds += 1
                if ball_seen:
                    dx = float(fields[0]) * length - ball_x
                    dy = float(fields[1]) * width - ball_y
                    distance = dx * dx + dy * dy
                    team = teams[slot]
                    if distance < closest[team]:
                        closest[team], closest_speed[team] = distance, speed

            if ball_seen:
                nearest[frame] = math.sqrt(closest[HOME]), math.sqrt(closest[AWAY])
                nearest_speed[frame] = closest_speed
            if n_speeds:
                crowd_speed[frame] = total_speed / n_speeds
    return ball, nearest, nearest_speed, crowd_speed


def ramp(values, full, zero):
    """1 at `full` and beyond, 0 at `zero` and beyond, a straight line between. NaN stays NaN."""
    return np.clip((values - zero) / float(full - zero), 0.0, 1.0)


def rolling_mean(values, window):
    """Centred mean over +/- `window` frames, ignoring NaN. NaN where the whole window is NaN.

    Computed with cumulative sums, like baseline.smooth, so the cost does not grow with the window.
    """
    seen = ~np.isnan(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(seen)))

    index = np.arange(len(values))
    low = np.maximum(0, index - window)
    high = np.minimum(len(values), index + window + 1)
    n_seen = counts[high] - counts[low]

    out = np.full(len(values), np.nan)
    has_data = n_seen > 0
    out[has_data] = (totals[high] - totals[low])[has_data] / n_seen[has_data]
    return out


def ball_speed(ball, lag):
    """Speed in m/s over +/- `lag` frames. NaN unless the ball is seen at both ends."""
    moved = ball[2 * lag:] - ball[:-2 * lag]
    out = np.full(len(ball), np.nan)
    out[lag:-lag] = np.hypot(moved[:, 0], moved[:, 1]) * FPS / (2.0 * lag)
    return out


def dead_ball_points(ball, crowd_speed, pitch):
    """(n,) how much each frame looks like a dead ball, from 0 to 1.

    Quiet players and a quiet ball have to agree. Either one alone happens in open play: players
    walk while a centre-back passes it around, and the ball sits still at a goalkeeper's feet.
    A ball that is not seen counts as quiet, because players walking while the ball has gone out of
    shot is exactly what a stoppage looks like. While players run, a lost ball earns nothing here.
    """
    calm = np.nan_to_num(ramp(rolling_mean(crowd_speed, DEAD_WINDOW), WALKING_SPEED, PLAYING_SPEED),
                          nan=1.0)
    ball_moving = ball_speed(ball, STILL_LAG) > STILL_SPEED
    quiet = 1.0 - rolling_mean(ball_moving.astype(np.float64), DEAD_WINDOW)

    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    outside = ((x < -OFF_PITCH_MARGIN) | (x > length + OFF_PITCH_MARGIN)
               | (y < -OFF_PITCH_MARGIN) | (y > width + OFF_PITCH_MARGIN))
    off_pitch = rolling_mean(np.where(np.isnan(x), np.nan, outside), OFF_PITCH_WINDOW) > 0.5

    return np.maximum(calm * quiet, off_pitch.astype(np.float64))


def frame_points(ball, nearest, nearest_speed, crowd_speed, pitch):
    """(n, 3) points for H, A and D on one half, from the tracking alone.

    Only the gap between the two teams earns anything, so a contested ball earns nothing and stays
    with the last team, which is how CHALLENGE.md defines it. A ball whose speed is unknown, at the
    edge of a gap in the tracking, gets the benefit of the doubt rather than being called flying.
    A player standing over a dead ball is not in control of it, so dead-ball evidence scales this
    down: that is what stops the taker of a free kick from earning points before the kick.
    """
    closeness = ramp(nearest, CONTROL_NEAR, CONTROL_FAR)
    faster = ball_speed(ball, SPEED_LAG)[:, np.newaxis] - np.nan_to_num(nearest_speed)
    at_feet = np.nan_to_num(ramp(faster, CARRIED_EXTRA, FLYING_EXTRA), nan=1.0)
    control = np.nan_to_num(closeness * at_feet)

    dead = dead_ball_points(ball, crowd_speed, pitch)
    alive = 1.0 - dead
    edge = (control[:, HOME] - control[:, AWAY]) * alive

    points = np.zeros((len(ball), 3))
    points[:, HOME] = alive + np.maximum(edge, 0.0)
    points[:, AWAY] = alive + np.maximum(-edge, 0.0)
    points[:, DEAD] = dead
    return points


def tag_points(events, clubs, n_frames, blocks):
    """(n_frames, 3) points from the Smart Tagging.

    Team names come from the ATD metadata rather than being spelled out here, and every throw-in is
    tagged N/A (DATA.md), so a set piece without a team still argues for a dead ball but gives
    nobody the restart. Phases overlap, so they vote once per frame rather than adding up.
    """
    sides = {club.strip().lower(): side for club, side in zip(clubs, (HOME, AWAY))}
    points = np.zeros((n_frames, 3))
    in_possession = np.zeros((n_frames, 2), dtype=bool)
    n_matched = 0

    for event in events:
        code = (event["code"] or "").strip().lower()
        kind = (event["tags"].get("Type") or "").strip().lower()
        side = sides.get((event["tags"].get("Team") or "").strip().lower())
        start, stop = event["start_frame"], event["end_frame"]
        if side is not None:
            n_matched += 1

        if code == SET_PIECES_CODE:
            restart = start + int(RESTART_POSITION * (stop - start))
            points[start:restart, DEAD] += SET_PIECE_POINTS
            if side is not None:
                points[restart:stop + AFTER_RESTART, side] += SET_PIECE_POINTS
            if kind == KICKOFF_TYPE:
                # The ball cannot be alive before the first whistle of a half. The tag says when
                # that is, so no frame number is needed here.
                for _, first, last in blocks:
                    if first <= stop <= last:
                        points[first:start, DEAD] += BEFORE_KICKOFF_POINTS
        elif side is not None and code in IN_POSSESSION_CODES:
            in_possession[start:stop, side] = True
        elif side is not None and code in OUT_OF_POSSESSION_CODES:
            in_possession[start:stop, 1 - side] = True

    if not n_matched:
        print("WARNING: no Smart Tagging team matches {}, so team tags are being ignored".format(
            " or ".join(clubs)))
    points[:, :DEAD] += PHASE_POINTS * in_possession
    return points


def best_labels(points, costs=COSTS):
    """The best label for every frame of one half: the Viterbi path. Returns (n,) H, A or D codes.

    points is (n, 3): how much each frame argues for H, A and D. costs[a][b] is the price of a
    change from label a to label b. The answer is the run of labels with the most points in total
    once every change has paid its price. possession_v2.py and possession_model.py use this same
    function with their own costs.

    How it works, in plain words: go through the frames once. For each label, keep the best total
    that any run of labels ending in that label could have by now, and note which label that run
    had on the frame before. At the end, start from the best label on the last frame and follow
    those notes backwards. This finds the best run without trying every possible one.

    When two answers score the same, the change is made as late as possible: possession stays with
    the last team until the new one shows control, so a neutral stretch before a change belongs to
    the old label. Plain lists, because small numpy arrays are slower in a loop this tight.
    """
    rows = points.tolist()
    states = range(len(costs))
    score = rows[0]  # best total so far for a run ending in each label
    came_from = [None] * len(rows)  # for each frame and label: the label on the frame before

    for frame in range(1, len(rows)):
        previous, score = score, [0.0] * len(costs)
        origins = []
        for state in states:
            # Staying in the same label is free; arriving from another one pays the change.
            best, origin = previous[state], state
            for other in states:
                if other != state and previous[other] - costs[other][state] >= best - TIE:
                    best, origin = previous[other] - costs[other][state], other
            score[state] = best + rows[frame][state]
            origins.append(origin)
        came_from[frame] = origins

    # Walk back from the best label on the last frame.
    path = np.zeros(len(rows), dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for frame in range(len(rows) - 1, 0, -1):
        path[frame - 1] = came_from[frame][path[frame]]
    return path


def build(game=GAME, data_dir=None):
    """Labels for the whole video. X outside the two halves, where nothing is graded."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)

    print("reading {} ...".format(os.path.basename(raw_path)))
    ball, nearest, nearest_speed, crowd_speed = read_tracking(
        raw_path, team_of_each_track(meta_path), n_frames, pitch)

    graded = np.concatenate([crowd_speed[first:last + 1] for _, first, last in blocks])
    if (np.isnan(graded).all()
            or not PLAUSIBLE_SPEED[0] <= np.nanmedian(graded) <= PLAUSIBLE_SPEED[1]):
        print("WARNING: player speeds do not look like m/s, so the dead-ball points are unreliable")

    tags = tag_points(load_smart_tagging(game, data_dir), team_names(meta_path), n_frames, blocks)

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        print("picking labels for half {} ...".format(half))
        stop = last + 1
        points = tags[first:stop] + frame_points(ball[first:stop], nearest[first:stop],
                                                 nearest_speed[first:stop], crowd_speed[first:stop],
                                                 pitch)
        labels[first:stop] = LABELS[best_labels(points)]
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    path = write_submission(os.path.join(args.out, OUT_NAME), build())
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("Now check it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("Then score it: python evaluation/run_evaluation.py --no-detail")
