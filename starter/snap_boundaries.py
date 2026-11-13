"""Move each change of possession to the touch that made it.

Viterbi places a change of team where the evidence has built up enough to pay the switch cost, which
is usually a little after the touch that really changed it. spellF1 averages over IoU 0.3 to 0.7, so
a possession that is right but drawn half a second late still loses the strict thresholds. Here,
after Viterbi, every change into a team (from the other team, and optionally from a dead ball) looks
up to WINDOW frames either way for the frame where the ball changes velocity most sharply with a
player of the new team within reach, and moves the boundary there if that touch is clear enough.
Boundaries never cross a neighbouring one, so the number of possessions does not change.

The settings are chosen on the tuning matches' out-of-fold second-look answers, then scored once on
the held-back matches, as for every other choice (RULES.md). Nothing here reads the public minutes.

    python starter/snap_boundaries.py tune
    python starter/snap_boundaries.py test
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    sys.path.insert(0, folder)
import possession_metrics as pm  # noqa: E402
import possession_v2  # noqa: E402
from possession_v1 import ramp  # noqa: E402

WINDOWS = (12, 25, 50)  # frames either way: 0.5, 1 and 2 s
THRESHOLDS = (0.2, 0.4, 0.6)  # touch strength x reach needed to move a boundary
REACHES = (2.0, 3.0)  # metres at which a player no longer counts as touching the ball
RESTARTS = (False, True)  # also move dead ball -> team changes
SETTINGS_FILE = "snap.json"


def team_nearest(ball, xy, team):
    """(n, 2) distance from the ball to the nearest tracked player of each team, NaN if none."""
    team = np.broadcast_to(team, xy.shape[:2])
    to_ball = np.hypot(xy[..., 0] - ball[:, 0:1], xy[..., 1] - ball[:, 1:2])
    out = np.full((len(ball), 2), np.nan)
    for side in (0, 1):
        reach = np.where((team == side) & ~np.isnan(to_ball), to_ball, np.inf).min(axis=1)
        out[np.isfinite(reach), side] = reach[np.isfinite(reach)]
    return out


def snap_half(codes, ball, xy, team, window, threshold, reach_far, restarts):
    """Move the boundaries of one half's labels in place. Returns how many moved."""
    touch = possession_v2.touch_strength(ball)
    reach = np.nan_to_num(ramp(team_nearest(ball, xy, team), possession_v2.TOUCH_NEAR, reach_far))
    evidence = touch[:, None] * reach  # (n, 2): a touch by each team at each frame
    bounds = (np.flatnonzero(codes[1:] != codes[:-1]) + 1).tolist()
    moved, floor = 0, 1
    for index, b in enumerate(bounds):
        old, new = int(codes[b - 1]), int(codes[b])
        ceiling = (bounds[index + 1] - 1) if index + 1 < len(bounds) else len(codes) - 1
        eligible = new in pm.POSSESSION_LABELS and (
            old in pm.POSSESSION_LABELS or (restarts and old == pm.D))
        if eligible:
            lo, hi = max(floor, b - window), min(ceiling, b + window)
            if hi > lo:
                scores = evidence[lo:hi + 1, new]
                best = float(scores.max())
                if best >= threshold:
                    # The strongest touch, nearest the Viterbi boundary on a tie.
                    candidates = np.flatnonzero(scores >= best - 1e-9) + lo
                    target = int(candidates[np.argmin(np.abs(candidates - b))])
                    if target < b:
                        codes[target:b] = new
                    elif target > b:
                        codes[b:target] = old
                    if target != b:
                        moved += 1
                    b = target
        floor = b + 1
    return moved


def snap(codes, blocks, ball, xy, team, window, threshold, reach_far, restarts):
    """A copy of whole-match codes with every half's boundaries moved to their touches."""
    out = codes.copy()
    team = np.asarray(team)
    for _, first, last in blocks:
        stop = last + 1
        half_team = team[first:stop] if team.ndim == 2 else team
        snap_half(out[first:stop], ball[first:stop], xy[first:stop], half_team,
                  window, threshold, reach_far, restarts)
    return out


def combos():
    return [dict(window=w, threshold=t, reach_far=r, restarts=s)
            for w, t, r, s in itertools.product(WINDOWS, THRESHOLDS, REACHES, RESTARTS)]


def stage_tune(train):
    with open(train.second_path()) as handle:
        second = json.load(handle)
    costs = train.model.cost_table(second["costs"]["switch"], second["costs"]["stop"],
                                   second["costs"]["restart"])
    items = train.load_checks(with_rules=False)
    cases = []
    for item in items:
        points = np.load(os.path.join(train.WORK_DIR,
                                      "oof2_{}_{}.npy".format(item["name"], item["seed"])))
        match = train.other.load_match(train.STEMS[item["name"]], train.other.DATA_DIR)
        ball, xy, _, team = train.as_arrays(train.other.damage(match, item["seed"]))
        cases.append((item, train.decode(points, item["blocks"], costs), ball, xy, team))
    print("{} match copies decoded".format(len(cases)), flush=True)

    def score(settings):
        rows = []
        for item, codes, ball, xy, team in cases:
            out = codes if settings is None else snap(codes, item["blocks"], ball, xy, team,
                                                     **settings)
            rows.append(train.score_levels(item["truths"], item["runs"], out))
        return np.mean(rows, axis=0)

    base = score(None)
    print("without snapping       {}  mean {:.4f}".format(
        " ".join("{:.4f}".format(m) for m in base), float(np.mean(base))), flush=True)
    results = []
    for settings in combos():
        means = score(settings)
        results.append((float(np.mean(means)), settings, means))
        print("{:<60} {}  mean {:.4f}".format(str(settings), " ".join(
            "{:.4f}".format(m) for m in means), float(np.mean(means))), flush=True)
    best_mean, chosen, means = max(results, key=lambda r: r[0])
    if best_mean <= float(np.mean(base)):
        chosen = None
    print("\nchosen: {} -> {} (without: {})".format(
        chosen, np.round(means, 4).tolist(), np.round(base, 4).tolist()))
    with open(os.path.join(train.WORK_DIR, SETTINGS_FILE), "w") as handle:
        json.dump(chosen, handle)


def stage_test(train):
    with open(os.path.join(train.WORK_DIR, SETTINGS_FILE)) as handle:
        chosen = json.load(handle)
    if chosen is None:
        print("tuning found no setting better than leaving the boundaries alone")
        return
    model_path, settings_path = train.trial_paths()
    booster, settings = train.model.load_model(model_path, settings_path)
    second_booster = train.model.load_second_model(settings, settings_path)
    first = train.model.cost_table(settings["switch_cost"], settings["stop_cost"],
                                   settings["restart_cost"])
    sc = settings["second_costs"]
    second_costs = train.model.cost_table(sc["switch"], sc["stop"], sc["restart"])
    print("held-back matches; snapping", chosen)
    scenarios = (("default damage", {}),
                 ("ball lost twice as often", {"LOSS_RATE": 2.0 * train.other.LOSS_RATE}),
                 ("ball found half as fast", {"FIND_RATE": 0.5 * train.other.FIND_RATE}),
                 ("many wrong-team tracks", {"WRONG_TEAM": 0.20}))
    for title, damage in scenarios:
        rows = {"+ second look": [], "+ snapped": []}
        for name in train.HELD_BACK:
            seed = train.other.SEED + train.other.MATCHES.index(train.STEMS[name])
            match, broken, points, _ = train.run_on(booster, None, name, seed, damage)
            looked = train.second_points(second_booster, points, match["blocks"], first)
            codes = train.decode(looked, match["blocks"], second_costs)
            ball, xy, _, team = train.as_arrays(broken)
            snapped = snap(codes, match["blocks"], ball, xy, team, **chosen)
            runs = pm.valid_runs(match["truth"])
            truths = [train.labels_at(match["truth"], level) for level in train.LEVELS]
            rows["+ second look"].append(train.score_levels(truths, runs, codes))
            rows["+ snapped"].append(train.score_levels(truths, runs, snapped))
        print("\n" + title)
        print("  {:<15} {}    mean".format("", " ".join(
            "{:>6}".format("{:.0f}s".format(l) if l else "given") for l in train.LEVELS)))
        for key, values in rows.items():
            means = np.mean(values, axis=0)
            print("  {:<15} {}  {:.4f}".format(key, " ".join("{:>6.3f}".format(m) for m in means),
                                               float(np.mean(means))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("tune", "test"))
    args = parser.parse_args()
    import train_possession_model as train
    {"tune": stage_tune, "test": stage_test}[args.stage](train)
