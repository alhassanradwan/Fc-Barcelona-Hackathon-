"""Train the possession model on public matches, choose its Viterbi costs, and test it.

    python starter/train_possession_model.py features   # damaged copies of the tuning matches
    python starter/train_possession_model.py cv         # hold out each tuning match in turn
    python starter/train_possession_model.py costs      # choose the costs on those predictions
    python starter/train_possession_model.py final      # train on all four, save to models/
    python starter/train_possession_model.py test       # score on the three held-back matches

None of the competition's data is read here. Training uses IDSSE matches only, as described in
test_on_other_matches.py, so the only labels the model has seen are from other games (RULES.md).

The split is by match. Frames from one match are so alike that a random split would report a score
far above what the model can really do. J03WN1, J03WOY and J03WR9 are never used for training or for
choosing anything in this file.

Why every copy is damaged differently
-------------------------------------
The real tracking has errors nobody has measured here, so a model trained on one fixed kind of damage
could learn that damage rather than football. Each training copy draws its own damage from
DAMAGE_RANGES, wide on purpose, so the model cannot lean on any one pattern, such as a missing ball
always meaning a stoppage. The 'test' stage also scores kinds of damage outside the training mix.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    sys.path.insert(0, folder)
import possession_metrics as pm  # noqa: E402
import possession_model as model  # noqa: E402
import possession_v1  # noqa: E402
import possession_v2  # noqa: E402
import test_on_other_matches as other  # noqa: E402

# Params
TUNING = ("J03WMX", "J03WOH", "J03WPY", "J03WQQ")
HELD_BACK = ("J03WN1", "J03WOY", "J03WR9")
STEMS = {stem[-6:]: stem for stem in other.MATCHES}
WORK_DIR = os.path.join(other.DATA_DIR, "model_work")

COPIES_PER_MATCH = 3
EVERY_NTH_FRAME = 5  # neighbouring frames say almost the same thing
TRAINING_SEED = 40
CHECK_SEEDS = (11, 12)  # default damage, for comparing with possession_v2 when choosing costs
N_PROCESSES = 8

# (name in test_on_other_matches, lowest, highest), drawn afresh for every training copy.
DAMAGE_RANGES = (("BALL_MISSING", 0.30, 0.50), ("BALL_MISSING_DEAD", 0.55, 0.90),
                 ("BALL_GAP_LIVE_S", 1.0, 3.0), ("BALL_GAP_DEAD_S", 5.0, 20.0),
                 ("BALL_EDGE_BLUR_S", 1.0, 6.0), ("BALL_NOISE_M", 0.2, 0.8),
                 ("PLAYER_NOISE_M", 0.1, 0.5), ("SPEED_NOISE_MS", 0.1, 0.6),
                 ("BALL_DRIFT_M", 0.2, 1.0), ("PLAYER_DRIFT_M", 0.1, 0.6),
                 ("CAMERA_HEIGHT_M", 20.0, 45.0), ("WRONG_TEAM", 0.0, 0.08),
                 ("FALSE_BALL_RATE", 0.0, 0.005), ("PLAYER_MISSING", 0.03, 0.12))

LIGHTGBM = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l2": 5.0, "seed": 2027, "deterministic": True, "force_row_wise": True,
            "num_threads": 16, "verbosity": -1}
MOST_ROUNDS = 800
PATIENCE = 40
LOG_EVERY = 25

SWITCH_COSTS = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
STOP_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
RESTART_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
LEVELS = (0.0, 2.0, 5.0, 8.0)  # seconds of brief touches not counted, 0 being the labels as given
# The public five minutes keep spells of about a second, so these two count for the choice.
CHOICE_LEVELS = (0.0, 2.0)


def as_arrays(match):
    """(ball, xy, speed, team) for the frames of the match."""
    return match["ball"], match["xy"], match["speed"], match["team_of"]


def randomised_damage(match, seed):
    """test_on_other_matches.damage with its settings drawn from DAMAGE_RANGES."""
    rng = np.random.RandomState(seed)
    keep = {name: getattr(other, name) for name, _, _ in DAMAGE_RANGES}
    for name, low, high in DAMAGE_RANGES:
        setattr(other, name, rng.uniform(low, high))
    try:
        return other.damage(match, seed)
    finally:
        for name, value in keep.items():
            setattr(other, name, value)


def training_copy(job):
    """One damaged copy of one match as sampled feature rows, saved to disk. Teams both ways round."""
    name, copy = job
    path = os.path.join(WORK_DIR, "train_{}_{}.npz".format(name, copy))
    if os.path.isfile(path):
        return path

    match = other.load_match(STEMS[name], other.DATA_DIR)
    seed = TRAINING_SEED + 100 * other.MATCHES.index(STEMS[name]) + copy
    broken = randomised_damage(match, seed)
    ball, xy, speed, team = as_arrays(broken)
    rows, labels = [], []
    for _, first, last in match["blocks"]:
        stop = last + 1
        offset = (seed + first) % EVERY_NTH_FRAME
        truth = match["truth"][first:stop][offset::EVERY_NTH_FRAME]
        for flip in (False, True):
            sides = 1 - team[first:stop] if flip else team[first:stop]
            features, names = model.frame_features(ball[first:stop], xy[first:stop], speed[first:stop],
                                                   sides, match["pitch"])
            rows.append(features[offset::EVERY_NTH_FRAME])
            labels.append(np.where(truth == pm.D, pm.D, 1 - truth) if flip else truth)
    np.savez(path, X=np.concatenate(rows), y=np.concatenate(labels), names=np.array(names))
    return path


def stage_features():
    os.makedirs(WORK_DIR, exist_ok=True)
    jobs = [(name, copy) for name in TUNING for copy in range(COPIES_PER_MATCH)]
    start = time.time()
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for count, path in enumerate(pool.imap_unordered(training_copy, jobs), start=1):
            print("{}/{} {} ({:.0f} s)".format(count, len(jobs), os.path.basename(path),
                                               time.time() - start), flush=True)


def load_rows(names):
    rows, labels, feature_names = [], [], None
    for name in names:
        for copy in range(COPIES_PER_MATCH):
            with np.load(os.path.join(WORK_DIR, "train_{}_{}.npz".format(name, copy))) as saved:
                rows.append(saved["X"])
                labels.append(saved["y"])
                feature_names = [str(n) for n in saved["names"]]
    return np.concatenate(rows), np.concatenate(labels), feature_names


def check_points(booster, name, seed):
    """Log-probabilities for the whole of one match under the default damage, plus the truth."""
    match = other.load_match(STEMS[name], other.DATA_DIR)
    broken = other.damage(match, seed)
    ball, xy, speed, team = as_arrays(broken)
    points = np.zeros((len(ball), 3), dtype=np.float32)
    for _, first, last in match["blocks"]:
        stop = last + 1
        points[first:stop] = model.predict_points(booster, ball[first:stop], xy[first:stop],
                                                  speed[first:stop], team[first:stop], match["pitch"])
    return points, match, broken


def stage_cv():
    """Train without each tuning match in turn, and keep the predictions for the one left out."""
    import lightgbm
    results = []
    for left_out in TUNING:
        start = time.time()
        train_names = [name for name in TUNING if name != left_out]
        X, y, names = load_rows(train_names)
        X_valid, y_valid, _ = load_rows([left_out])
        train = lightgbm.Dataset(X, y, feature_name=names, free_raw_data=True)
        valid = lightgbm.Dataset(X_valid, y_valid, reference=train, free_raw_data=True)
        booster = lightgbm.train(LIGHTGBM, train, num_boost_round=MOST_ROUNDS, valid_sets=[valid],
                                 callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False),
                                            lightgbm.log_evaluation(LOG_EVERY)])
        results.append(booster.best_iteration)
        print("without {}: {} rows, best round {}, log loss {:.4f} ({:.0f} s)".format(
            left_out, len(y), booster.best_iteration, booster.best_score["valid_0"]["multi_logloss"],
            time.time() - start), flush=True)
        for seed in CHECK_SEEDS:
            points, _, _ = check_points(booster, left_out, seed)
            np.save(os.path.join(WORK_DIR, "oof_{}_{}.npy".format(left_out, seed)), points)
    with open(os.path.join(WORK_DIR, "rounds.json"), "w") as handle:
        json.dump(results, handle)
    print("best rounds:", results)


def score_levels(truths, runs, codes):
    """spellF1 against each version of the truth."""
    return [pm.spell_scores(truth, codes, runs)["mean_f1"] for truth in truths]


def load_checks():
    """The tuning matches under default damage, with their out-of-fold points and possession_v2."""
    items = []
    for name in TUNING:
        match = other.load_match(STEMS[name], other.DATA_DIR)
        truths = [match["truth"] if not level else other.absorb_touches(match["truth"], level)
                  for level in LEVELS]
        runs = pm.valid_runs(match["truth"])
        for seed in CHECK_SEEDS:
            broken = other.damage(match, seed)
            items.append({"name": name, "truths": truths, "runs": runs, "blocks": match["blocks"],
                          "points": np.load(os.path.join(WORK_DIR, "oof_{}_{}.npy".format(name, seed))),
                          "rules": other.run_method(possession_v2, broken)})
    return items


def decode(points, blocks, costs):
    codes = np.full(len(points), pm.X, dtype=np.int8)
    for _, first, last in blocks:
        codes[first:last + 1] = model.best_labels(points[first:last + 1].astype(np.float64), costs)
    return codes


def stage_costs():
    """Choose the Viterbi costs on predictions for matches the model did not train on."""
    items = load_checks()
    choice = [LEVELS.index(level) for level in CHOICE_LEVELS]

    def evaluate(switch, stop, restart):
        costs = model.cost_table(switch, stop, restart)
        rows, counts = [], []
        for item in items:
            codes = decode(item["points"], item["blocks"], costs)
            rows.append(score_levels(item["truths"], item["runs"], codes))
            counts.append(len(pm.possession_spells(codes, item["runs"])))
        means = np.mean(rows, axis=0)
        return float(means[choice].mean()), means, float(np.mean(counts))

    reference = np.mean([score_levels(i["truths"], i["runs"], i["rules"]) for i in items], axis=0)
    print("possession_v2, same matches and damage: {}".format(np.round(reference, 3).tolist()))

    settings = {"switch": 24.0, "stop": 32.0, "restart": 32.0}
    best = evaluate(settings["switch"], settings["stop"], settings["restart"])[0]
    print("{:>7} {:>6} {:>7} | {}  {:>6} {:>5}".format(
        "switch", "stop", "restart", " ".join("{:>5.0f}s".format(l) for l in LEVELS), "choice", "poss"))
    for key, values in (("switch", SWITCH_COSTS), ("stop", STOP_COSTS), ("restart", RESTART_COSTS),
                        ("switch", SWITCH_COSTS)):
        for value in values:
            trial = dict(settings, **{key: value})
            score, means, count = evaluate(trial["switch"], trial["stop"], trial["restart"])
            print("{:>7.0f} {:>6.0f} {:>7.0f} | {}  {:>6.4f} {:>5.0f}".format(
                trial["switch"], trial["stop"], trial["restart"],
                " ".join("{:>6.3f}".format(m) for m in means), score, count), flush=True)
            if score > best:
                best, settings = score, trial
        print("  best so far: {} -> {:.4f}".format(settings, best), flush=True)
    with open(os.path.join(WORK_DIR, "costs.json"), "w") as handle:
        json.dump(settings, handle)
    print("chosen:", settings)


def stage_final():
    """Train on all four tuning matches and save the model and its settings."""
    import lightgbm
    with open(os.path.join(WORK_DIR, "rounds.json")) as handle:
        rounds = int(round(np.mean(json.load(handle))))
    with open(os.path.join(WORK_DIR, "costs.json")) as handle:
        costs = json.load(handle)
    X, y, names = load_rows(TUNING)
    start = time.time()
    train = lightgbm.Dataset(X, y, feature_name=names)
    booster = lightgbm.train(LIGHTGBM, train, num_boost_round=rounds, valid_sets=[train],
                             callbacks=[lightgbm.log_evaluation(LOG_EVERY)])
    os.makedirs(os.path.dirname(model.MODEL_PATH), exist_ok=True)
    booster.save_model(model.MODEL_PATH)
    with open(model.SETTINGS_PATH, "w") as handle:
        json.dump({"switch_cost": costs["switch"], "stop_cost": costs["stop"],
                   "restart_cost": costs["restart"], "rounds": rounds,
                   "trained_on": list(TUNING), "held_back": list(HELD_BACK),
                   "copies_per_match": COPIES_PER_MATCH, "every_nth_frame": EVERY_NTH_FRAME,
                   "feature_names": names}, handle, indent=2)
    print("trained {} rounds on {} rows in {:.0f} s".format(rounds, len(y), time.time() - start))
    print("model file: {:.1f} MB".format(os.path.getsize(model.MODEL_PATH) / 1e6))
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gains)[:20]
    print("features that mattered most:")
    for index in order:
        print("  {:<20} {:>5.1f}%".format(names[index], 100.0 * gains[index] / gains.sum()))


def run_on(booster, costs, name, seed, damage_settings=None):
    """Codes from the model, possession_v2 and possession_v1 on one held-back match."""
    match = other.load_match(STEMS[name], other.DATA_DIR)
    keep = {}
    for key, value in (damage_settings or {}).items():
        keep[key] = getattr(other, key)
        setattr(other, key, value)
    try:
        broken = other.damage(match, seed)
    finally:
        for key, value in keep.items():
            setattr(other, key, value)
    ball, xy, speed, team = as_arrays(broken)
    points = np.zeros((len(ball), 3))
    for _, first, last in match["blocks"]:
        stop = last + 1
        points[first:stop] = model.predict_points(booster, ball[first:stop], xy[first:stop],
                                                  speed[first:stop], team[first:stop], match["pitch"])
    return match, broken, points, {"model": decode(points, match["blocks"], costs),
                                   "possession_v2": other.run_method(possession_v2, broken),
                                   "possession_v1": other.run_method(possession_v1, broken)}


def stage_test():
    """Score on the held-back matches: the default damage, then damage outside the training mix."""
    booster, settings = model.load_model()
    costs = model.cost_table(settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    print("costs:", settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    scenarios = (("default damage", {}),
                 ("ball missing as often in play as in stoppages", {"BALL_MISSING_DEAD": 0.41}),
                 ("ball missing far more in stoppages", {"BALL_MISSING_DEAD": 0.95}),
                 ("very noisy positions", {"BALL_NOISE_M": 1.5, "PLAYER_NOISE_M": 1.0}),
                 ("many wrong-team tracks", {"WRONG_TEAM": 0.20}),
                 ("many false ball detections", {"FALSE_BALL_RATE": 0.015}),
                 ("ball missing much more often", {"BALL_MISSING": 0.60, "BALL_MISSING_DEAD": 0.90}),
                 ("no ball at all", {"BALL_MISSING": 0.999, "BALL_MISSING_DEAD": 0.999,
                                     "BALL_GAP_LIVE_S": 60.0, "BALL_GAP_DEAD_S": 60.0}))
    for title, damage_settings in scenarios:
        rows = {"model": [], "possession_v2": [], "possession_v1": []}
        macro = {key: [] for key in rows}
        counts = {key: [] for key in rows}
        per_match = {key: [] for key in rows}
        for name in HELD_BACK:
            seed = other.SEED + other.MATCHES.index(STEMS[name])
            match, _, _, codes = run_on(booster, costs, name, seed, damage_settings)
            truth = match["truth"]
            runs = pm.valid_runs(truth)
            truths = [truth if not level else other.absorb_touches(truth, level) for level in LEVELS]
            for key, predicted in codes.items():
                scores = score_levels(truths, runs, predicted)
                rows[key].append(scores)
                per_match[key].append(scores[0])
                macro[key].append(pm.macro_f1(pm.scored_confusion(pm.confusion(truth, predicted))))
                counts[key].append(len(pm.possession_spells(predicted, runs)))
        print("\n{}".format(title))
        print("  {:<15} {}  {:>7} {:>5}   as given, per match".format(
            "", " ".join("{:>6}".format("{:.0f}s".format(l) if l else "given") for l in LEVELS),
            "macroF1", "poss"))
        for key in rows:
            print("  {:<15} {}  {:>7.4f} {:>5.0f}   {}".format(
                key, " ".join("{:>6.3f}".format(m) for m in np.mean(rows[key], axis=0)),
                np.mean(macro[key]), np.mean(counts[key]),
                " ".join("{:.3f}".format(s) for s in per_match[key])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("features", "cv", "costs", "final", "test"))
    args = parser.parse_args()
    {"features": stage_features, "cv": stage_cv, "costs": stage_costs, "final": stage_final,
     "test": stage_test}[args.stage]()
