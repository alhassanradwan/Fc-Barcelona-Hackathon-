"""Possession v2: possession_v1 with sharper evidence of who has the ball, tuned on other matches.

In short
--------
The same three steps as possession_v1 (measure, give points, pick the labels with Viterbi), and the
same code for reading the tracking, the dead-ball points and the Smart Tagging. Two things are new:

  - Only a clear gap between the teams counts. The nearer team has to be nearer by CLEAR_MARGIN
    metres to earn full points, so a duel earns little and a clean take-over shows up at once.
  - Touches count. When the ball suddenly changes speed or direction next to a player, that player
    touched it. This catches a one-touch interception, which never looks like control.

The cost of a change of label is the same as in possession_v1: the gain comes from better evidence.
possession_model.py uses this file's evidence (frame_points, touch_strength) as some of its
features, so the model starts from what these rules already know.

    pip install -r requirements.txt          # numpy and pandas, nothing else
    python starter/possession_v2.py          # writes evaluation/results/possession_v2.csv
    python evaluation/validate_submission.py evaluation/results/possession_v2.csv
    python evaluation/run_evaluation.py --no-detail

Expect a few minutes: it reads 199,319 lines of tracking.

Where the values come from
--------------------------
Every value under Params was picked on four public IDSSE matches (J03WMX, J03WOH, J03WPY, J03WQQ)
and checked on three others (J03WN1, J03WOY, J03WR9). None came from this competition's ground
truth, public or private. starter/test_on_other_matches.py reproduces the check.

Why the cost was chosen on four versions of the labels: the IDSSE labels change team far more often
than CHALLENGE.md's definition suggests. A lower cost scored higher on them as given, but lost
ground when brief touches were not counted, as the test on the other three matches showed. So the
cost was chosen by its average over four versions of the labels: as given, and with spells under 2,
5 and 8 seconds handed back to the rival on both sides. A setting that suits only one of them would
be a bet on how the real ground truth was made.

The dead-ball values stay as they are in possession_v1, because tuning them here would mostly fit
the simulated damage rather than football. No frame number of this match is written anywhere in
this file, for the reason RULES.md gives.

Known limits
------------
Those of possession_v1. It still misses most possessions shorter than about five seconds.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from baseline import (AWAY, HOME, LABELS, RESULTS_DIR, team_names,  # noqa: E402
                      team_of_each_track, write_submission)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402
from possession_v1 import (DEAD, FPS, PLAUSIBLE_SPEED, ball_speed, dead_ball_points,  # noqa: E402
                           ramp, read_tracking, tag_points)
from possession_v1 import best_labels as viterbi  # noqa: E402

# Params
OUT_NAME = "possession_v2.csv"

# Control. Full points with the ball this close to a player, none at the second distance, in metres.
CONTROL_NEAR, CONTROL_FAR = 1.0, 2.5
# How much faster than that player the ball may be moving, in m/s, measured over +/- SPEED_LAG
# frames.
CARRIED_EXTRA, FLYING_EXTRA = 4.0, 10.0
SPEED_LAG = 4
# The nearer team gets full points only if it is nearer by this many metres.
CLEAR_MARGIN = 1.0

# Touches. Velocity is measured over TOUCH_LAG frames either side of a frame, long enough that
# tracking noise does not look like a kick. No points below TOUCH_LOW m/s of change, full points at
# TOUCH_HIGH.
TOUCH_POINTS = 2.0
TOUCH_LAG = 8
TOUCH_LOW, TOUCH_HIGH = 10.0, 14.0
# How close a player has to be to have made the touch, in metres, from full points to none.
TOUCH_NEAR, TOUCH_FAR = 1.0, 2.0

# What a change of label costs, the same as possession_v1. One frame of clear evidence is worth
# about one point, and a brief spell by the other team pays twice, to change and to change back.
SWITCH_COST = 12.5
STOP_COST = 25.0
RESTART_COST = 25.0
COSTS = ((0.0, SWITCH_COST, STOP_COST),
         (SWITCH_COST, 0.0, STOP_COST),
         (RESTART_COST, RESTART_COST, 0.0))


def touch_strength(ball):
    """(n,) how sharply the ball changes velocity at each frame, 0 to 1. 0 if it is not known."""
    velocity = (ball[TOUCH_LAG:] - ball[:-TOUCH_LAG]) * FPS / float(TOUCH_LAG)
    difference = velocity[TOUCH_LAG:] - velocity[:-TOUCH_LAG]
    change = np.full(len(ball), np.nan)
    change[TOUCH_LAG:-TOUCH_LAG] = np.hypot(difference[:, 0], difference[:, 1])
    return np.nan_to_num(ramp(change, TOUCH_HIGH, TOUCH_LOW))


def frame_points(ball, nearest, nearest_speed, crowd_speed, pitch):
    """(n, 3) points for H, A and D on one half, from the tracking alone.

    possession_v1 scores the difference in closeness, which stays small while both teams are near
    the ball. Here the nearer team earns its own control, scaled by how clearly it is the nearer
    one, and a touch adds to that. Dead-ball evidence scales it all down, for the same reason as in
    possession_v1: standing over a dead ball is not control of it.
    """
    closeness = ramp(nearest, CONTROL_NEAR, CONTROL_FAR)
    faster = ball_speed(ball, SPEED_LAG)[:, np.newaxis] - np.nan_to_num(nearest_speed)
    at_feet = np.nan_to_num(ramp(faster, CARRIED_EXTRA, FLYING_EXTRA), nan=1.0)
    control = np.nan_to_num(closeness * at_feet)

    home_lead = np.nan_to_num(nearest[:, AWAY] - nearest[:, HOME])  # positive: home is nearer
    nearer = np.where(home_lead >= 0, HOME, AWAY)
    edge = (np.clip(home_lead / CLEAR_MARGIN, -1.0, 1.0)
            * control[np.arange(len(ball)), nearer])

    within_reach = np.nan_to_num(ramp(nearest, TOUCH_NEAR, TOUCH_FAR))
    edge = edge + (TOUCH_POINTS * touch_strength(ball)
                   * (within_reach[:, HOME] - within_reach[:, AWAY]))

    dead = dead_ball_points(ball, crowd_speed, pitch)
    alive = 1.0 - dead
    edge = edge * alive

    points = np.zeros((len(ball), 3))
    points[:, HOME] = alive + np.maximum(edge, 0.0)
    points[:, AWAY] = alive + np.maximum(-edge, 0.0)
    points[:, DEAD] = dead
    return points


def best_labels(points):
    """The Viterbi path for one half with this file's COSTS (see possession_v1.best_labels)."""
    return viterbi(points, COSTS)


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
