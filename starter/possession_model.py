"""Possession model: our submission. Two LightGBM models and Viterbi label every frame.

In short
--------
For every frame of the Barcelona match this file says who has the ball (H, A or D, and X outside
the two halves) and writes evaluation/results/possession_model.csv. Three steps:

  1. Features. For every frame, 99 numbers that describe what the tracking shows: where each team
     stands and how it moves, where the ball is and how fast it goes, how long the ball has been
     missing from the tracking, what possession_v2's rules make of the frame, and where the players
     stand at restarts (throw-ins, corners, kick-offs).
  2. Two models. The first LightGBM model reads the features and gives the chance of H, A and D on
     every frame. The second, smaller one (the "second look") reads the first one's answers around
     each frame and corrects its usual mistakes.
  3. Viterbi. The same algorithm as possession_v1 joins those chances into whole possessions:
     every change of label costs points, so one odd frame cannot flip the answer.

Both models were trained only on other matches (train_possession_model.py). Nothing is trained or
tuned here, and nothing was ever tuned on this match.

    pip install -r requirements-model.txt
    python starter/possession_model.py       # writes evaluation/results/possession_model.csv
    python evaluation/validate_submission.py evaluation/results/possession_model.csv
    python evaluation/run_evaluation.py --no-detail

Expect a few minutes. Use Python 3.11, which the models were trained and tested on.

Why a model, and not only rules
-------------------------------
possession_v2's rules can only use the ball, so on the 41% of graded frames where the ball is
missing from the tracking they have no evidence at all. A model can still use what is visible then:
how each team is shaped and moving, how long the ball has been gone, where it was before and after
the gap, and where the players stand when play restarts: someone at a touchline for a throw-in or
at a corner flag for a corner, players in the box, both teams in their own halves for a kick-off.
possession_v2's own evidence goes in as features too, so the model starts from what the rules
already know.

Why LightGBM
------------
The features are a big table of numbers with many gaps (NaN wherever the ball or a player is not
tracked). LightGBM learns fast from large tables, handles missing values without any filling-in,
and can be made deterministic, so training it again gives exactly the same model.

Why a second look
-----------------
The first model judges one frame at a time and keeps making the same kinds of mistake: a flicker to
the other team for a moment, a change of team drawn a little late. The second look reads the first
model's answers up to 2 s either way, and how long its pick and its Viterbi labels have held and
will hold, so it can see those patterns and correct them. It learned from the first model's answers
on matches that model had not seen, so it learned the first model's real mistakes.

Why Viterbi stays
-----------------
Even after the second look the answers can flicker, and spellF1 scores whole possessions, not single
frames. Viterbi's points are the log of the probabilities, so the run of labels it picks is the most
likely one once each change of label has paid its cost. The costs were chosen on matches the models
had not trained on, because a model is over-confident on its own training data.

Why the two teams are treated alike
-----------------------------------
Every half is turned so the home team attacks towards larger x (worked out from where each team's
deepest players stand), so the model never has to learn which end is which. The model then predicts
twice, once with the teams swapped, and averages the two. So H and A mean nothing on their own: the
model judges only what each team is doing.

Why Smart Tagging stays out of the models
-----------------------------------------
The training matches have no Smart Tagging, and tags invented for them would teach the models to
trust tags too much. So the tags are added only at the Viterbi step, weighed against the cost of a
change of team the same way as in possession_v2.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (ROOT, HERE):
    sys.path.insert(0, folder)
import possession_v2  # noqa: E402
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, LABELS, MISSING, RESULTS_DIR,  # noqa: E402
                      team_names, team_of_each_track, write_submission)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402
from possession_v1 import (DEAD, FPS, PLAUSIBLE_SPEED, ball_speed, ramp, rolling_mean,  # noqa: E402
                           tag_points)
from possession_v1 import best_labels as viterbi  # noqa: E402

# Params
OUT_NAME = "possession_model.csv"
# The trained first model, and its settings: costs, feature names, and which second look to use.
MODEL_PATH = os.path.join(ROOT, "models", "possession_lgbm.txt")
SETTINGS_PATH = os.path.join(ROOT, "models", "possession_lgbm.json")
SMALLEST_PROBABILITY = 1e-4  # so that one frame can never cost more than a few points
# The restart clues (restart_features).
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # metres: the penalty box, from its goal line and middle
RESTART_WINDOW = 50  # frames, 2 s: how near the touchline and a corner flag before and after
# The second look (second_features).
SECOND_FILE = "possession_second.txt"  # the second look, kept beside the settings that use it
SECOND_OFFSETS = (-50, -25, -12, -6, 6, 12, 25, 50)  # frames: 2, 1, 0.5 and 0.25 s either way
SECOND_WINDOWS = (12, 25, 50, 125)  # frames averaged before and after
SECOND_SPANS = (12, 25)  # frames either way for the lowest and highest edge
LONGEST_RUN = 750  # frames, 30 s: runs longer than this all look the same to the second look
SWAP = [AWAY, HOME, DEAD]  # H, A, D columns with the teams swapped


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


def restart_features(x, y, tracked, team, pitch):
    """The features frame_features adds last, by name: who restarts play, from where the players
    stand. A throw-in is taken at a touchline, a corner at a corner flag, a free kick into the box
    with players waiting in it, and a kick-off with both teams in their own halves. x and y are
    turned as frame_features turns them, so the home team attacks towards larger x."""
    length, width = pitch
    out, side_gaps, corner_gaps, own_halves = {}, [], [], []
    for side, tag in ((HOME, "h"), (AWAY, "a")):
        mine = tracked & (team == side)
        players = mine.sum(axis=1).astype(np.float64)
        to_side = np.where(mine, np.minimum(y, width - y), np.inf).min(axis=1)
        end = length if side == HOME else 0.0  # the goal line this team attacks
        to_corner = np.where(mine, np.minimum(np.hypot(x - end, y), np.hypot(x - end, y - width)),
                             np.inf).min(axis=1)
        deep = (x > length - BOX_DEPTH) if side == HOME else (x < BOX_DEPTH)
        in_box = mine & deep & (np.abs(y - width / 2.0) < BOX_HALF_WIDTH)
        at_home = mine & ((x < length / 2.0) if side == HOME else (x > length / 2.0))
        out["side_gap_" + tag] = np.where(np.isfinite(to_side), to_side, np.nan)
        out["corner_gap_" + tag] = np.where(np.isfinite(to_corner), to_corner, np.nan)
        out["in_box_" + tag] = in_box.sum(axis=1).astype(np.float64)
        out["own_half_" + tag] = at_home.sum(axis=1) / np.where(players > 0, players, np.nan)
        side_gaps.append(out["side_gap_" + tag])
        corner_gaps.append(out["corner_gap_" + tag])
        own_halves.append(out["own_half_" + tag])
    side_any, corner_any = np.fmin(*side_gaps), np.fmin(*corner_gaps)
    out["side_gap_before"] = window_mean(side_any, RESTART_WINDOW, 0)
    out["side_gap_after"] = window_mean(side_any, 0, RESTART_WINDOW)
    out["corner_gap_before"] = window_mean(corner_any, RESTART_WINDOW, 0)
    out["corner_gap_after"] = window_mean(corner_any, 0, RESTART_WINDOW)
    out["both_own_half"] = np.fmin(*own_halves)
    return out


def frame_features(ball, xy, speed, team, pitch):
    """(n, n_features) float32 for one half, and the feature names. team is (n, p), 0 or 1.

    In order: each team's shape, movement and distance to the ball; the ball itself; what
    possession_v2's rules make of the frame; the gap in the ball tracking the frame sits in; the
    same evidence averaged over windows before and after; and last, the restart clues. Names ending
    in _h and _a belong to one team each, so swapping team gives the swapped features.
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

    # Who restarts play after the ball goes out, from where the players stand.
    out.update(restart_features(x, y, tracked, team, pitch))

    names = list(out)
    return np.column_stack([out[name] for name in names]).astype(np.float32), names


def team_features(ball, xy, speed, team, pitch):
    """(direct, swapped): the features for one half, as it is and with the teams swapped.

    The model is asked both ways round and the two answers are averaged (points_from), so it judges
    each team by what that team does, never by whether it is the home or the away side."""
    return (frame_features(ball, xy, speed, team, pitch)[0],
            frame_features(ball, xy, speed, 1 - team, pitch)[0])


def points_from(booster, direct, swapped):
    """(n, 3) log-probabilities of H, A and D, averaged over swapping the teams."""
    # The swapped answer's H and A columns are swapped back before the two are averaged.
    average = (booster.predict(direct) + booster.predict(swapped)[:, [AWAY, HOME, DEAD]]) / 2.0
    return np.log(np.clip(average, SMALLEST_PROBABILITY, 1.0))


def predict_points(booster, ball, xy, speed, team, pitch):
    """(n, 3) log-probabilities of H, A and D for one half, averaged over swapping the teams."""
    return points_from(booster, *team_features(ball, xy, speed, team, pitch))


def check_feature_count(booster, features):
    """Stop with a clear message if the saved model reads a different number of features than this
    code makes. That happens when the features changed but the model was not trained again."""
    if features.shape[1] != booster.num_feature():
        raise ValueError("the saved model reads {} features, but this code makes {}: train it "
                         "again (see train_possession_model.py)".format(booster.num_feature(),
                                                                        features.shape[1]))


def shifted(values, offset):
    """values moved so that frame t holds frame t + offset, NaN past either end."""
    out = np.full(len(values), np.nan)
    if offset > 0:
        out[:-offset] = values[offset:]
    elif offset < 0:
        out[-offset:] = values[:offset]
    else:
        out[:] = values
    return out


def run_facts(labels):
    """For each frame: (frames since its run of one label began, frames until it ends, the label of
    the run before, the label of the run after), -1 where there is none."""
    n = len(labels)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [n]))
    run_of = np.repeat(np.arange(len(starts)), stops - starts)
    index = np.arange(n)
    run_labels = labels[starts].astype(np.int64)
    last = len(starts) - 1
    before = np.where(run_of > 0, run_labels[np.maximum(run_of - 1, 0)], -1)
    after = np.where(run_of < last, run_labels[np.minimum(run_of + 1, last)], -1)
    return index - starts[run_of], stops[run_of] - 1 - index, before, after


def second_features(points, first_costs):
    """(n, n_features) float32 for one half, and the names: what the second look reads, from the
    first model's (n, 3) log-probabilities. first_costs are the first model's own, for its Viterbi
    labels. points[:, SWAP] gives the features with the teams swapped."""
    log_h, log_a, log_d = (points[:, k].astype(np.float64) for k in (HOME, AWAY, DEAD))
    edge = log_h - log_a
    dead = np.exp(log_d)
    out = {"log_h": log_h, "log_a": log_a, "log_d": log_d}
    for offset in SECOND_OFFSETS:
        out["edge_{:+d}".format(offset)] = shifted(edge, offset)
        out["dead_{:+d}".format(offset)] = shifted(dead, offset)
    for window in SECOND_WINDOWS:
        out["edge_before_{}".format(window)] = window_mean(edge, window, 0)
        out["edge_after_{}".format(window)] = window_mean(edge, 0, window)
        out["dead_before_{}".format(window)] = window_mean(dead, window, 0)
        out["dead_after_{}".format(window)] = window_mean(dead, 0, window)
    for span in SECOND_SPANS:
        out["edge_low_{}".format(span)] = minimum_filter1d(edge, 2 * span + 1, mode="nearest")
        out["edge_high_{}".format(span)] = maximum_filter1d(edge, 2 * span + 1, mode="nearest")
    for tag, labels in (("pick", np.argmax(points, axis=1)),
                        ("path", best_labels(points.astype(np.float64), first_costs))):
        since, until, before, after = run_facts(labels)
        out[tag] = labels
        out[tag + "_since"] = np.minimum(since, LONGEST_RUN)
        out[tag + "_until"] = np.minimum(until, LONGEST_RUN)
        out[tag + "_length"] = np.minimum(since + until + 1, 2 * LONGEST_RUN)
        out[tag + "_before"] = before
        out[tag + "_after"] = after
    names = list(out)
    return np.column_stack([out[name] for name in names]).astype(np.float32), names


def second_look(second_booster, points, first_costs):
    """(n, 3) the second look's log-probabilities for one half, from the first model's, averaged
    over swapping the teams as the first model's are."""
    points = np.asarray(points, dtype=np.float32)  # as training saved the ones it learned from
    direct = second_features(points, first_costs)[0]
    swapped = second_features(points[:, SWAP], first_costs)[0]
    average = (second_booster.predict(direct) + second_booster.predict(swapped)[:, SWAP]) / 2.0
    return np.log(np.clip(average, SMALLEST_PROBABILITY, 1.0))


def best_labels(points, costs):
    """The Viterbi path for one half (possession_v1.best_labels), with the costs passed in, because
    the right costs depend on the model that made the points: they come from its settings."""
    return viterbi(points, costs)


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


def load_model(model_path=MODEL_PATH, settings_path=SETTINGS_PATH):
    """(booster, settings). A readable error if the training has not been run."""
    if not os.path.isfile(model_path) or not os.path.isfile(settings_path):
        raise IOError("no trained model at {}.\nRun python starter/train_possession_model.py "
                      "first, or restore it from the repository.".format(model_path))
    import lightgbm
    with open(settings_path) as handle:
        settings = json.load(handle)
    return lightgbm.Booster(model_file=model_path), settings


def load_second_model(settings, settings_path=SETTINGS_PATH):
    """The second look these settings name, kept beside them, or None if they name none."""
    if not settings.get("second_model"):
        return None
    path = os.path.join(os.path.dirname(settings_path), settings["second_model"])
    if not os.path.isfile(path):
        raise IOError("the settings at {} use {}, which is missing".format(settings_path, path))
    import lightgbm
    return lightgbm.Booster(model_file=path)


def build(game=GAME, data_dir=None):
    """Labels for the whole video: H, A or D in the two halves, X outside them (never graded)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)

    # The two models and their Viterbi costs. The second look reads the first model's answers with
    # the first model's own costs, and its own answers are joined with the costs chosen for it.
    booster, settings = load_model()
    second_booster = load_second_model(settings)
    first_costs = cost_table(settings["switch_cost"], settings["stop_cost"],
                             settings["restart_cost"])
    costs = first_costs
    if second_booster is not None:
        chosen = settings["second_costs"]
        costs = cost_table(chosen["switch"], chosen["stop"], chosen["restart"])
        print("with the second look, costs {:g} {:g} {:g}".format(
            chosen["switch"], chosen["stop"], chosen["restart"]))

    print("reading {} ...".format(os.path.basename(raw_path)))
    team_of_track = team_of_each_track(meta_path)
    ball, xy, speed = read_full_tracking(raw_path, team_of_track, n_frames, pitch, blocks)

    graded = np.concatenate([speed[first:last + 1].ravel() for _, first, last in blocks])
    if (np.isnan(graded).all()
            or not PLAUSIBLE_SPEED[0] <= np.nanmedian(graded) <= PLAUSIBLE_SPEED[1]):
        print("WARNING: player speeds do not look like m/s, so the dead-ball features are "
              "unreliable")

    # Smart Tagging joins only at the Viterbi step. It is weighed against the switch cost Viterbi
    # pays here exactly as possession_v2 weighs it against its own (see the module docstring).
    tags = tag_points(load_smart_tagging(game, data_dir), team_names(meta_path), n_frames, blocks)
    tag_scale = costs[HOME][AWAY] / possession_v2.SWITCH_COST  # the switch cost Viterbi pays here

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        print("predicting half {} ...".format(half))
        stop = last + 1
        team = np.broadcast_to(team_of_track, (stop - first, len(team_of_track)))
        # Step 1: the features, as they are and with the teams swapped.
        direct, swapped = team_features(ball[first:stop], xy[first:stop].astype(np.float64),
                                        speed[first:stop].astype(np.float64), team, pitch)
        check_feature_count(booster, direct)
        # Step 2: the first model, then the second look.
        points = points_from(booster, direct, swapped)
        if second_booster is not None:
            points = second_look(second_booster, points, first_costs).astype(np.float32)
        # Step 3: Viterbi joins the frames into possessions.
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
