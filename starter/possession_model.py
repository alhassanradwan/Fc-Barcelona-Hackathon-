"""Possession model: LightGBM predicts H, A or D for every frame, then Viterbi picks the labels.

The model was trained only on other matches (see train_possession_model.py) and is not trained here.
This file reads the tracking, turns it into features, asks the saved model, and writes the file.

    pip install -r requirements-model.txt
    python starter/possession_model.py       # writes evaluation/results/possession_model.csv
    python evaluation/validate_submission.py evaluation/results/possession_model.csv
    python evaluation/run_evaluation.py --no-detail

Expect a few minutes. It uses Python 3.11, because that is what the model was trained and tested on.

Why a model at all
------------------
possession_v2 can only use the ball, so on the 41% of graded frames without one it has no evidence.
The features here add what is still visible then: how each team is shaped and moving, how long the
ball has been gone, and where it was before and after the gap. possession_v2's own evidence goes in
as features too, so the model starts from what the rules already know.

Why Viterbi stays
-----------------
The model judges one frame at a time, helped by features that look a few seconds either way, so its
answers can still flicker. Viterbi joins them into whole possessions, which is what spellF1 scores.
Its costs are chosen on predictions for matches the model did not train on, because a model is
over-confident on its own training data.

Why the teams are treated alike
-------------------------------
Every half is turned so the home team attacks towards larger x, worked out from where each team's
deepest players stand, so the model never has to learn which end is which. The model then predicts
twice, once with the teams swapped, and averages. H and A therefore have no meaning of their own.

Smart Tagging stays out of the model. These training matches have none, and tags invented for them
would teach it to trust them too much. It is added at the Viterbi step with the same weight relative
to a team change as in possession_v2.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (ROOT, HERE):
    sys.path.insert(0, folder)
import possession_v2  # noqa: E402
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, LABELS, MISSING, RESULTS_DIR,  # noqa: E402
                      team_names, team_of_each_track, write_submission)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402
from possession_v1 import (DEAD, FPS, PLAUSIBLE_SPEED, TIE, ball_speed, ramp,  # noqa: E402
                           rolling_mean, tag_points)

# Params
OUT_NAME = "possession_model.csv"
MODEL_PATH = os.path.join(ROOT, "models", "possession_lgbm.txt")
SETTINGS_PATH = os.path.join(ROOT, "models", "possession_lgbm.json")
SMALLEST_PROBABILITY = 1e-4  # so that one frame can never cost more than a few points


def window_mean(values, before, after):
    """Mean over the frames from `before` ahead of each frame to `after` behind it, ignoring NaN."""
    seen = ~np.isnan(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - before)
    high = np.minimum(len(values), index + after + 1)
    n_seen = counts[high] - counts[low]
    out = np.full(len(values), np.nan)
    has_data = n_seen > 0
    out[has_data] = (totals[high] - totals[low])[has_data] / n_seen[has_data]
    return out


def last_known(values):
    """(value carried forward over NaN, frames since it was known). NaN before the first one."""
    index = np.arange(len(values))
    latest = np.maximum.accumulate(np.where(np.isnan(values), -1, index))
    known = latest >= 0
    carried = np.full(len(values), np.nan)
    since = np.full(len(values), np.nan)
    carried[known] = values[latest[known]]
    since[known] = (index - latest)[known]
    return carried, since


def next_known(values):
    """(value carried backward over NaN, frames until it is known). NaN after the last one."""
    carried, until = last_known(values[::-1])
    return carried[::-1], until[::-1]


def drift(values, lag):
    """Change per second across +/- lag frames."""
    out = np.full(len(values), np.nan)
    out[lag:-lag] = (values[2 * lag:] - values[:-2 * lag]) * FPS / (2.0 * lag)
    return out


def face_the_same_way(ball, xy, team, pitch):
    """(ball, xy) turned if needed so the home team attacks towards larger x in this half.

    Each team's deepest and furthest players stay nearer their own goal on average, the goalkeeper
    above all, so the midpoint of those two says which end a team defends.
    """
    x = xy[..., 0]
    tracked = ~np.isnan(x)
    middle = []
    for side in (HOME, AWAY):
        mine = tracked & (team == side)
        low = np.where(mine, x, np.inf).min(axis=1)
        high = np.where(mine, x, -np.inf).max(axis=1)
        usable = np.isfinite(low) & np.isfinite(high)
        middle.append(float(np.mean((low[usable] + high[usable]) / 2.0)) if usable.any() else 0.0)
    if middle[HOME] <= middle[AWAY]:
        return ball, xy
    far_corner = np.array(pitch, dtype=np.float64)
    return far_corner - ball, far_corner - xy


def frame_features(ball, xy, speed, team, pitch):
    """(n, n_features) float32 for one half, and the feature names. team is (n, p), 0 or 1.

    The names end in _h and _a for the two teams, so swapping team gives the swapped features.
    """
    length, width = pitch
    ball, xy = face_the_same_way(ball, xy, team, pitch)
    n_frames = len(ball)
    rows = np.arange(n_frames)
    x, y = xy[..., 0], xy[..., 1]
    ball_x, ball_y = ball[:, 0], ball[:, 1]
    tracked = ~np.isnan(x)
    timed = tracked & ~np.isnan(speed)
    seen = ~np.isnan(ball_x)
    to_ball = np.hypot(x - ball_x[:, np.newaxis], y - ball_y[:, np.newaxis])

    out = {}
    nearest = np.full((n_frames, 2), np.nan)
    nearest_speed = np.full((n_frames, 2), np.nan)
    means = {}
    for side, tag in ((HOME, "h"), (AWAY, "a")):
        mine = tracked & (team == side)
        players = mine.sum(axis=1).astype(np.float64)
        per_player = np.where(players > 0, players, np.nan)
        mean_x = np.where(mine, x, 0.0).sum(axis=1) / per_player
        mean_y = np.where(mine, y, 0.0).sum(axis=1) / per_player
        low = np.where(mine, x, np.inf).min(axis=1)
        high = np.where(mine, x, -np.inf).max(axis=1)
        moving = timed & (team == side)
        per_timed = np.where(moving.sum(axis=1) > 0, moving.sum(axis=1), np.nan)
        top = np.where(moving, speed, -np.inf).max(axis=1)

        reach = np.where(mine & ~np.isnan(to_ball), to_ball, np.inf)
        closest = reach.argmin(axis=1)
        distance = reach[rows, closest]
        found = np.isfinite(distance)
        nearest[found, side] = distance[found]
        nearest_speed[found, side] = speed[rows, closest][found]
        means[side] = mean_x

        out["players_" + tag] = players
        out["mean_x_" + tag] = mean_x
        out["mean_y_" + tag] = mean_y
        out["spread_x_" + tag] = np.sqrt(np.maximum(
            0.0, (np.where(mine, x, 0.0) ** 2).sum(axis=1) / per_player - mean_x ** 2))
        out["spread_y_" + tag] = np.sqrt(np.maximum(
            0.0, (np.where(mine, y, 0.0) ** 2).sum(axis=1) / per_player - mean_y ** 2))
        out["low_x_" + tag] = np.where(np.isfinite(low), low, np.nan)
        out["high_x_" + tag] = np.where(np.isfinite(high), high, np.nan)
        out["speed_" + tag] = np.where(moving, speed, 0.0).sum(axis=1) / per_timed
        out["top_speed_" + tag] = np.where(np.isfinite(top), top, np.nan)
        for radius in (2.0, 5.0, 10.0):
            out["within_{:.0f}_{}".format(radius, tag)] = np.where(
                seen, (reach < radius).sum(axis=1), np.nan)
        out["near_" + tag] = nearest[:, side]
        out["near_speed_" + tag] = nearest_speed[:, side]
        out["ball_from_mean_" + tag] = ball_x - mean_x
        smooth_x = rolling_mean(mean_x, 12)
        out["drift_2s_" + tag] = drift(smooth_x, 25)
        out["drift_6s_" + tag] = drift(smooth_x, 75)
        out["speed_5s_" + tag] = rolling_mean(out["speed_" + tag], 62)

    n_timed = timed.sum(axis=1)
    crowd_speed = np.where(timed, speed, 0.0).sum(axis=1) / np.where(n_timed > 0, n_timed, np.nan)
    home_lead = nearest[:, AWAY] - nearest[:, HOME]  # positive: home is the nearer team
    out["lead"] = home_lead
    out["team_gap_x"] = means[HOME] - means[AWAY]

    out["ball_x"] = ball_x
    out["ball_y"] = ball_y
    out["ball_to_side"] = np.minimum(ball_y, width - ball_y)
    out["ball_to_end"] = np.minimum(ball_x, length - ball_x)
    out["ball_speed_2"] = ball_speed(ball, 2)
    out["ball_speed_4"] = ball_speed(ball, 4)
    out["ball_speed_12"] = ball_speed(ball, 12)
    out["ball_drift_x"] = drift(ball_x, 4)
    out["ball_drift_y"] = drift(ball_y, 4)

    # What possession_v2 already knows, so the model starts from the rules.
    rule_points = possession_v2.frame_points(ball, nearest, nearest_speed, crowd_speed, pitch)
    edge = rule_points[:, HOME] - rule_points[:, AWAY]
    dead = rule_points[:, DEAD]
    touch = possession_v2.touch_strength(ball)
    within_reach = np.nan_to_num(ramp(nearest, possession_v2.TOUCH_NEAR, possession_v2.TOUCH_FAR))
    touch_edge = touch * (within_reach[:, HOME] - within_reach[:, AWAY])
    out["edge"] = edge
    out["dead"] = dead
    out["touch"] = touch
    out["touch_edge"] = touch_edge

    # The gap in the ball tracking this frame sits in, and what is known either side of it.
    last_x, since_seen = last_known(ball_x)
    next_x, until_seen = next_known(ball_x)
    last_y, _ = last_known(ball_y)
    next_y, _ = next_known(ball_y)
    last_lead, _ = last_known(home_lead)
    next_lead, _ = next_known(home_lead)
    out["since_seen"] = np.minimum(since_seen / FPS, 30.0)
    out["until_seen"] = np.minimum(until_seen / FPS, 30.0)
    out["gap_length"] = np.minimum((since_seen + until_seen) / FPS, 60.0)
    out["seen_1s"] = rolling_mean(seen.astype(np.float64), 25)
    out["seen_5s"] = rolling_mean(seen.astype(np.float64), 125)
    out["last_lead"] = last_lead
    out["next_lead"] = next_lead
    out["last_ball_x"] = last_x
    out["next_ball_x"] = next_x
    out["gap_move"] = np.hypot(next_x - last_x, next_y - last_y)

    # The same evidence over a range of windows, centred and looking each way.
    for window in (12, 25, 50, 125):
        out["edge_around_{}".format(window)] = rolling_mean(edge, window)
    for window in (25, 50, 125):
        out["edge_before_{}".format(window)] = window_mean(edge, window, 0)
        out["edge_after_{}".format(window)] = window_mean(edge, 0, window)
    for window in (12, 25, 50):
        out["touch_before_{}".format(window)] = window_mean(touch_edge, window, 0)
        out["touch_after_{}".format(window)] = window_mean(touch_edge, 0, window)
    for window in (25, 50):
        out["lead_around_{}".format(window)] = rolling_mean(home_lead, window)
    for window in (25, 125):
        out["dead_around_{}".format(window)] = rolling_mean(dead, window)
        out["crowd_around_{}".format(window)] = rolling_mean(crowd_speed, window)
    out["crowd_before_50"] = window_mean(crowd_speed, 50, 0)
    out["crowd_after_50"] = window_mean(crowd_speed, 0, 50)
    out["speed_gap_5s"] = out["speed_5s_h"] - out["speed_5s_a"]

    names = list(out)
    return np.column_stack([out[name] for name in names]).astype(np.float32), names


def predict_points(booster, ball, xy, speed, team, pitch):
    """(n, 3) log-probabilities of H, A and D for one half, averaged over swapping the teams."""
    features, _ = frame_features(ball, xy, speed, team, pitch)
    direct = booster.predict(features)
    features, _ = frame_features(ball, xy, speed, 1 - team, pitch)
    swapped = booster.predict(features)
    average = (direct + swapped[:, [AWAY, HOME, DEAD]]) / 2.0
    return np.log(np.clip(average, SMALLEST_PROBABILITY, 1.0))


def best_labels(points, costs):
    """The state for each frame with the most points in total, after paying costs. The Viterbi path.

    The same as possession_v2.best_labels, with the costs passed in, because they depend on the
    model that produced the points. On a tie a change is made as late as possible.
    """
    rows = points.tolist()
    states = range(len(costs))
    score = rows[0]
    came_from = [None] * len(rows)

    for frame in range(1, len(rows)):
        previous, score = score, [0.0] * len(costs)
        origins = []
        for state in states:
            best, origin = previous[state], state
            for other in states:
                if other != state and previous[other] - costs[other][state] >= best - TIE:
                    best, origin = previous[other] - costs[other][state], other
            score[state] = best + rows[frame][state]
            origins.append(origin)
        came_from[frame] = origins

    path = np.zeros(len(rows), dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for frame in range(len(rows) - 1, 0, -1):
        path[frame - 1] = came_from[frame][path[frame]]
    return path


def cost_table(switch, stop, restart):
    """COSTS in the layout best_labels expects: rows are from, columns are to, in H, A, D order."""
    return ((0.0, switch, stop), (switch, 0.0, stop), (restart, restart, 0.0))


def read_full_tracking(raw_path, team_of_track, n_frames, pitch, blocks):
    """(ball, xy, speed) over every slot, in metres and m/s, for the frames inside the halves only.

    possession_v2's reader keeps only what it needs about each frame. The model needs every player.
    float32 keeps this to about 100 MB across 46 slots.
    """
    length, width = pitch
    n_slots = len(team_of_track)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    ball = np.full((n_frames, 2), np.nan)
    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not graded[frame]:
                continue
            if parts[2][0] != MISSING:
                fields = parts[2].split(",")
                ball[frame] = float(fields[0]) * length, float(fields[1]) * width
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                fields = group.split(",")
                xy[frame, slot] = float(fields[0]) * length, float(fields[1]) * width
                speed[frame, slot] = float(fields[2])
    return ball, xy, speed


def load_model():
    """(booster, settings). A readable error if the training has not been run."""
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(SETTINGS_PATH):
        raise IOError("no trained model at {}.\nRun python starter/train_possession_model.py "
                      "first, or restore it from the repository.".format(MODEL_PATH))
    import lightgbm
    with open(SETTINGS_PATH) as handle:
        settings = json.load(handle)
    return lightgbm.Booster(model_file=MODEL_PATH), settings


def build(game=GAME, data_dir=None):
    """Labels for the whole video. X outside the two halves, where nothing is graded."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)
    booster, settings = load_model()
    costs = cost_table(settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])

    print("reading {} ...".format(os.path.basename(raw_path)))
    team_of_track = team_of_each_track(meta_path)
    ball, xy, speed = read_full_tracking(raw_path, team_of_track, n_frames, pitch, blocks)

    graded = np.concatenate([speed[first:last + 1].ravel() for _, first, last in blocks])
    if (np.isnan(graded).all()
            or not PLAUSIBLE_SPEED[0] <= np.nanmedian(graded) <= PLAUSIBLE_SPEED[1]):
        print("WARNING: player speeds do not look like m/s, so the dead-ball features are unreliable")

    tags = tag_points(load_smart_tagging(game, data_dir), team_names(meta_path), n_frames, blocks)
    tag_scale = settings["switch_cost"] / possession_v2.SWITCH_COST

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        print("predicting half {} ...".format(half))
        stop = last + 1
        team = np.broadcast_to(team_of_track, (stop - first, len(team_of_track)))
        points = predict_points(booster, ball[first:stop], xy[first:stop].astype(np.float64),
                                speed[first:stop].astype(np.float64), team, pitch)
        labels[first:stop] = LABELS[best_labels(points + tag_scale * tags[first:stop], costs)]
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
