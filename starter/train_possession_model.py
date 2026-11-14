"""Train the possession model on other matches, choose its settings, and test it honestly.

In short
--------
possession_model.py needs two trained LightGBM models and a few settings. This file makes them from
the seven public IDSSE matches (see test_on_other_matches.py), and from nothing of the competition's
match. The work is split into stages, run one after the other:

    python starter/train_possession_model.py features   # 1. training rows from damaged copies
    python starter/train_possession_model.py cv         # 2. first model, each match left out
    python starter/train_possession_model.py costs      # 3. Viterbi costs for the first model
    python starter/train_possession_model.py second     # 4. the second look, and its costs
    python starter/train_possession_model.py final      # 5. both models, on all training matches
    python starter/train_possession_model.py test       # 6. score them on the held-back matches

Without --all, stages 1-5 use the four tuning matches, and 'test' scores the result on the three
held-back ones. Those chose nothing, so 'test' shows honestly whether the method works. With --all,
the same stages use all seven matches, and 'final --all' writes the model to submit into models/:

    python starter/train_possession_model.py features --all    # then cv, costs, second, final --all

Each stage keeps its work in a folder beside the repository (WORK_DIR, or WORK_DIR_ALL with --all),
so a later stage can be run again without redoing the earlier ones. Expect about 20 minutes for
stages 1-5, about 45 with --all, and about 20 for 'test'. Every step is deterministic: run again,
it gives exactly the same model.

Why other matches, and why their tracking is damaged first
-----------------------------------------------------------
RULES.md allows training on other matches in any amount, but the five public minutes of this one
are only a check. The IDSSE tracking is clean, though, and the competition's is not, so every
training match is first damaged to break the way the competition's tracking breaks
(test_on_other_matches.damage). Each of the three copies of a match gets its own random damage,
drawn from ranges around the measured settings, so the model learns football rather than one
pattern of damage (such as 'a missing ball always means a stoppage'). 'test' also scores kinds of
damage outside the training mix.

Why the matches are split whole
-------------------------------
Frames of one match are so alike that a random split of frames would report a score far above what
the model can really do. So whole matches are left out. 'cv' trains without each tuning match in
turn and predicts the one left out ("out-of-fold" answers), and every later choice is made on
those. J03WN1, J03WOY and J03WR9 are never used to train or choose anything, except with --all.

What is chosen, and how
-----------------------
- How many trees: 'cv' stops adding trees once the match left out stops improving (early
  stopping), and 'final' uses the average over the folds.
- The Viterbi costs: 'costs' tries a grid of costs for a change of team (switch), for the ball
  going dead (stop) and for play restarting (restart) on the out-of-fold answers, and keeps the
  best spellF1 on the labels as given. Costs within TIED_WITHIN of the best count as tied, and the
  tie goes to whichever does best over all four versions of the labels (LEVELS), because nobody
  knows how the real answers treat brief touches.
- The second look: 'second' trains it the same way, each match left out in turn, on the first
  model's out-of-fold answers, and chooses its own costs by the same search.

Which labels the model learns
-----------------------------
The labels as given: read frame by frame from the tracking's ball owner (BallPossession), with
nothing smoothed, the way CHALLENGE.md says the Barcelona answers were made. 'test' also scores
with brief touches not counted: spells under 2, 5 or 8 s between two of the rival's are handed back
to the rival (the 2s, 5s and 8s columns).

What we kept, and what it gained
--------------------------------
Every idea was tested the same way, with its rule set before running: first on the tuning matches,
each left out in turn, and only if it passed there, once on the held-back matches. These passed:
- The second look: on the held-back matches, +0.007 spellF1 as given and +0.008 to +0.011 at 2, 5
  and 8 s (measured before the restart clues were added).
- The restart clues (restart_features in possession_model.py): +0.008 as given and +0.006, +0.005
  and +0.004 at 2, 5 and 8 s on the tuning matches; on the held-back matches, over the default
  damage and three harder kinds, +0.014 as given and +0.009, +0.005 and +0.005.

What we tried and dropped
-------------------------
- Learning labels with brief touches handed back: lower on the held-back matches as given (0.539
  against 0.594) and at 2 s (0.646 against 0.660).
- A third model that makes a change of team cheaper where one looks likely: no gain (0.594 as
  given, with it and without it).
- Averaging models trained with several random seeds: +0.002, inside the noise.
- Training again on the model's own confident answers (self-training): -0.001.
- Facts about the tracking given to the second look: +0.004, under the bar of +0.005.
- A third look, reading the second look's answers: -0.004.
- Training the second look on more kinds of damage: -0.0004.
- Clues for goal kicks and for free-kick walls: -0.006 and -0.001 on the tuning matches.
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
# The matches, by the last six characters of their IDSSE ids.
TUNING = ("J03WMX", "J03WOH", "J03WPY", "J03WQQ")  # every choice is made on these
HELD_BACK = ("J03WN1", "J03WOY", "J03WR9")  # only for the last, honest check in 'test'
TRAIN = TUNING  # all seven with --all
STEMS = {stem[-6:]: stem for stem in other.MATCHES}
# Where the stages keep their work: beside the repository, so nothing large can be committed.
WORK_DIR = os.path.join(other.DATA_DIR, "model_work")
WORK_DIR_ALL = os.path.join(other.DATA_DIR, "model_work_all")

# The training rows.
COPIES_PER_MATCH = 3  # damaged copies of each match, each one damaged differently
EVERY_NTH_FRAME = 5  # neighbouring frames say almost the same thing, so one in five is enough
TRAINING_SEED = 40
CHECK_SEEDS = (11, 12)  # two copies of each match under the default damage, for choosing costs
N_PROCESSES = 8  # training copies made at the same time

# The damage of each training copy, drawn afresh for every copy: (name in test_on_other_matches,
# lowest, highest), as multiples of the measured setting there, then as plain values.
DAMAGE_SCALES = (("LOSS_RATE", 0.6, 1.6), ("LOSS_DEAD", 0.6, 1.6), ("FIND_RATE", 0.6, 1.6),
                 ("FIND_DEAD", 0.6, 1.6), ("GLITCH_RATE", 0.5, 2.0), ("BALL_DRIFT_M", 0.5, 1.5),
                 ("BALL_SMOOTH_FRAMES", 0.5, 1.5), ("PLAYER_SMOOTH_FRAMES", 0.5, 1.5),
                 ("SPEED_NOISE_MS", 0.5, 2.0), ("CAMERA_HEIGHT_M", 0.7, 1.5),
                 ("PLAYER_LOSS_RATE", 0.6, 1.6), ("GROUP_LOSS", 0.5, 1.5), ("GHOST_RATE", 0.0, 2.0),
                 ("BALL_EDGE_BLUR_S", 0.33, 2.0), ("OUTSIDE_M", 0.5, 2.0))
DAMAGE_VALUES = (("FILL_S", 3.0, 6.0), ("FILL_TRUE", 0.0, 0.6), ("WRONG_TEAM", 0.0, 0.08),
                 ("BALL_NOISE_M", 0.0, 0.03), ("PLAYER_NOISE_M", 0.0, 0.03),
                 ("PLAYER_DRIFT_M", 0.0, 0.3), ("FALSE_BALL_RATE", 0.0, 0.001))
# How much the ball's situation changes the chance of losing or finding it, as a power of the
# measured factors: 1 as measured, below 1 weaker, above stronger. The ATD's camera stands on one
# side, so which side of the pitch loses the ball more is swapped on half the copies.
SITUATION_FACTORS = ("LOSS_BY_SPEED", "FIND_BY_SPEED", "LOSS_ACROSS", "LOSS_ALONG", "LOSS_IN_CROWD",
                     "LOSS_IN_BOX")
SITUATION_POWER = (0.6, 1.4)
SIDED = ("LOSS_ACROSS", "LOSS_ALONG")

# The first model. Many small steps (learning_rate); leaves that need at least 500 rows each, so no
# leaf can learn one moment of one match; and every tree sees a random 70% of the rows and of the
# features, so the model leans on many features rather than a few. A fixed seed, a fixed number of
# threads and 'deterministic' make training again give exactly the same model.
LIGHTGBM = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l2": 5.0, "seed": 2027, "deterministic": True, "force_row_wise": True,
            "num_threads": 16, "verbosity": -1}
MOST_ROUNDS = 800  # the most trees 'cv' tries
PATIENCE = 40  # 'cv' stops once the match left out has not improved for this many trees
LOG_EVERY = 25

# The second look: the same settings with smaller trees, because it reads fewer features (51), all
# of one kind (the first model's answers).
SECOND_LIGHTGBM = dict(LIGHTGBM, num_leaves=31)

# The Viterbi costs that 'costs' and 'second' try: where the search starts, what it tries in turn
# (the switch cost again at the end, because its best value can move once the others are chosen),
# and how it prints them: (name, width, decimals).
SWITCH_COSTS = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
STOP_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
RESTART_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
START_COSTS = {"switch": 24.0, "stop": 32.0, "restart": 32.0}
COST_SEARCH = (("switch", SWITCH_COSTS), ("stop", STOP_COSTS), ("restart", RESTART_COSTS),
               ("switch", SWITCH_COSTS))
COST_COLUMNS = (("switch", 7, 0), ("stop", 6, 0), ("restart", 7, 0))
# Costs closer than this to the best spellF1 on the labels as given count as tied with it, and the
# tie goes to whichever does best over all of LEVELS. 0.002 is about ten possessions out of the
# several thousand in the check copies, far below what EVALUATION.md calls noise.
TIED_WITHIN = 0.002
LEVELS = (0.0, 2.0, 5.0, 8.0)  # seconds of brief touches not counted, 0 being the labels as given


def as_arrays(match):
    """(ball, xy, speed, team) for the frames of the match."""
    return match["ball"], match["xy"], match["speed"], match["team_of"]


def labels_at(truth, seconds):
    """truth with spells under `seconds` between two of the rival's handed back to the rival. With
    0, the labels as given."""
    return other.absorb_touches(truth, seconds) if seconds else truth


def trial_paths():
    """Where 'final' without --all keeps its model and settings: the ones 'test' scores."""
    return (os.path.join(WORK_DIR, "possession_lgbm.txt"),
            os.path.join(WORK_DIR, "possession_lgbm.json"))


def costs_path():
    """Where 'costs' keeps the first model's Viterbi costs."""
    return os.path.join(WORK_DIR, "costs.json")


def second_path():
    """Where 'second' keeps what it chose for the second look."""
    return os.path.join(WORK_DIR, "second.json")


# Stage 1: features --------------------------------------------------------------------------------

def randomised_damage(match, seed):
    """test_on_other_matches.damage with its settings drawn at random from DAMAGE_SCALES,
    DAMAGE_VALUES and SITUATION_POWER, then put back as they were."""
    rng = np.random.RandomState(seed)
    names = ([name for name, _, _ in DAMAGE_SCALES + DAMAGE_VALUES]
             + [name for name in SITUATION_FACTORS])
    keep = {name: getattr(other, name) for name in names}
    for name, low, high in DAMAGE_SCALES:
        setattr(other, name, keep[name] * rng.uniform(low, high))
    other.GROUP_LOSS = min(1.0, other.GROUP_LOSS)
    for name, low, high in DAMAGE_VALUES:
        setattr(other, name, rng.uniform(low, high))
    power = rng.uniform(*SITUATION_POWER)
    for name in SITUATION_FACTORS:
        factors = np.asarray(keep[name], dtype=np.float64) ** power
        if name in SIDED and rng.rand() < 0.5:
            factors = factors[::-1]
        setattr(other, name, tuple(factors) if factors.ndim else float(factors))
    try:
        return other.damage(match, seed)
    finally:
        for name, value in keep.items():
            setattr(other, name, value)


def training_copy(job):
    """Make one damaged copy of one match into training rows, and save them to disk.

    Every EVERY_NTH_FRAME-th frame of each half becomes two rows: as it is, and with the two teams
    swapped (its label H or A swapped with them). That doubles the rows and teaches the model that H
    and A mean nothing on their own. job is (match, copy number, folder): the folder comes with the
    job because the worker processes never see what the command line changed. A copy already on
    disk is kept, so the stage can be stopped and started again."""
    name, copy, work_dir = job
    path = os.path.join(work_dir, "train_{}_{}.npz".format(name, copy))
    if os.path.isfile(path):
        return path

    match = other.load_match(STEMS[name], other.DATA_DIR)
    seed = TRAINING_SEED + 100 * other.MATCHES.index(STEMS[name]) + copy
    broken = randomised_damage(match, seed)
    ball, xy, speed, team = as_arrays(broken)
    rows, labels = [], []
    for _, first, last in match["blocks"]:
        stop = last + 1
        offset = (seed + first) % EVERY_NTH_FRAME  # which frame in five, different for every copy
        truth = match["truth"][first:stop][offset::EVERY_NTH_FRAME]
        for flip in (False, True):
            sides = 1 - team[first:stop] if flip else team[first:stop]
            features, names = model.frame_features(ball[first:stop], xy[first:stop],
                                                   speed[first:stop], sides, match["pitch"])
            rows.append(features[offset::EVERY_NTH_FRAME])
            labels.append(np.where(truth == pm.D, pm.D, 1 - truth) if flip else truth)
    np.savez(path, X=np.concatenate(rows), y=np.concatenate(labels), names=np.array(names))
    return path


def stage_features():
    """Stage 1: the training rows of every damaged copy of every training match, several at once."""
    os.makedirs(WORK_DIR, exist_ok=True)
    jobs = [(name, copy, WORK_DIR) for name in TRAIN for copy in range(COPIES_PER_MATCH)]
    start = time.time()
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for count, path in enumerate(pool.imap_unordered(training_copy, jobs), start=1):
            print("{}/{} {} ({:.0f} s)".format(count, len(jobs), os.path.basename(path),
                                               time.time() - start), flush=True)


def load_rows(names):
    """(X, y, feature names): the saved training rows of these matches, every copy, in order."""
    rows, labels, feature_names = [], [], None
    for name in names:
        for copy in range(COPIES_PER_MATCH):
            with np.load(os.path.join(WORK_DIR, "train_{}_{}.npz".format(name, copy))) as saved:
                rows.append(saved["X"])
                labels.append(saved["y"])
                feature_names = [str(n) for n in saved["names"]]
    return np.concatenate(rows), np.concatenate(labels), feature_names


# Stage 2: cv --------------------------------------------------------------------------------------

def check_points(booster, name, seed):
    """The first model's log-probabilities for the whole of one match, damaged with this seed under
    the default damage. 0 outside the halves."""
    match = other.load_match(STEMS[name], other.DATA_DIR)
    broken = other.damage(match, seed)
    ball, xy, speed, team = as_arrays(broken)
    points = np.zeros((len(ball), 3), dtype=np.float32)
    for _, first, last in match["blocks"]:
        stop = last + 1
        points[first:stop] = model.predict_points(booster, ball[first:stop], xy[first:stop],
                                                  speed[first:stop], team[first:stop],
                                                  match["pitch"])
    return points


def stage_cv():
    """Stage 2: train the first model without each training match in turn, and save its answers for
    the match left out (out-of-fold). Every later choice is made on those answers."""
    import lightgbm
    results = []
    for left_out in TRAIN:
        start = time.time()
        train_names = [name for name in TRAIN if name != left_out]
        X, y, names = load_rows(train_names)
        X_valid, y_valid, _ = load_rows([left_out])
        train = lightgbm.Dataset(X, y, feature_name=names, free_raw_data=True)
        valid = lightgbm.Dataset(X_valid, y_valid, reference=train, free_raw_data=True)
        # Add trees while the match left out still improves (early stopping).
        booster = lightgbm.train(LIGHTGBM, train, num_boost_round=MOST_ROUNDS, valid_sets=[valid],
                                 callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False),
                                            lightgbm.log_evaluation(LOG_EVERY)])
        results.append(booster.best_iteration)
        print("without {}: {} rows, best round {}, log loss {:.4f} ({:.0f} s)".format(
            left_out, len(y), booster.best_iteration,
            booster.best_score["valid_0"]["multi_logloss"], time.time() - start), flush=True)
        for seed in CHECK_SEEDS:
            np.save(os.path.join(WORK_DIR, "oof_{}_{}.npy".format(left_out, seed)),
                    check_points(booster, left_out, seed))
    with open(os.path.join(WORK_DIR, "rounds.json"), "w") as handle:
        json.dump(results, handle)
    print("best rounds:", results)


# Stage 3: costs -----------------------------------------------------------------------------------

def score_levels(truths, runs, codes):
    """spellF1 against each version of the truth (see LEVELS)."""
    return [pm.spell_scores(truth, codes, runs)["mean_f1"] for truth in truths]


def load_checks(with_rules=True):
    """The training matches under the default damage (two copies each) with the first model's
    out-of-fold answers, and with_rules, what possession_v2 makes of them, for comparison."""
    items = []
    for name in TRAIN:
        match = other.load_match(STEMS[name], other.DATA_DIR)
        truths = [labels_at(match["truth"], level) for level in LEVELS]
        runs = pm.valid_runs(match["truth"])
        for seed in CHECK_SEEDS:
            item = {"name": name, "seed": seed, "truths": truths, "runs": runs,
                    "blocks": match["blocks"],
                    "points": np.load(os.path.join(WORK_DIR, "oof_{}_{}.npy".format(name, seed)))}
            if with_rules:
                item["rules"] = other.run_method(possession_v2, other.damage(match, seed))
            items.append(item)
    return items


def decode(points, blocks, costs):
    """Viterbi labels for every half of a match, X outside the halves."""
    codes = np.full(len(points), pm.X, dtype=np.int8)
    for _, first, last in blocks:
        stop = last + 1
        codes[first:stop] = model.best_labels(points[first:stop].astype(np.float64), costs)
    return codes


def scorer(items):
    """evaluate(trial) for search_costs: (spellF1 on the labels as given, spellF1 on each of LEVELS,
    possessions per match) for the items' points decoded with the trial's costs."""
    def evaluate(trial):
        costs = model.cost_table(trial["switch"], trial["stop"], trial["restart"])
        rows, counts = [], []
        for item in items:
            codes = decode(item["points"], item["blocks"], costs)
            rows.append(score_levels(item["truths"], item["runs"], codes))
            counts.append(len(pm.possession_spells(codes, item["runs"])))
        means = np.mean(rows, axis=0)
        return float(means[0]), means, float(np.mean(counts))
    return evaluate


def break_tie(trials, best):
    """(overall, settings) of the trial within TIED_WITHIN of best that does best over all LEVELS.

    trials holds (settings, score on the labels as given, mean score over all of LEVELS)."""
    tied = [(overall, settings) for settings, score, overall in trials
            if score >= best - TIED_WITHIN]
    return max(tied, key=lambda pair: pair[0])


def search_costs(evaluate, settings, search, columns):
    """The costs that do best on the labels as given, near-ties settled by break_tie.

    A simple search: starting from settings, try every value of one cost while the others stay put,
    keep the best, and go on to the next cost, in the order search lists them. Prints every
    trial."""
    best = evaluate(settings)[0]
    print("{} | {}  {:>6} {:>5}".format(
        " ".join("{:>{}}".format(key, width) for key, width, _ in columns),
        " ".join("{:>5.0f}s".format(l) for l in LEVELS), "choice", "poss"))
    trials = []
    for key, values in search:
        for value in values:
            trial = dict(settings, **{key: value})
            score, means, count = evaluate(trial)
            print("{} | {}  {:>6.4f} {:>5.0f}".format(
                " ".join("{:>{}.{}f}".format(trial[column], width, places)
                         for column, width, places in columns),
                " ".join("{:>6.3f}".format(m) for m in means), score, count), flush=True)
            trials.append((trial, score, float(np.mean(means))))
            if score > best:
                best, settings = score, trial
        print("  best so far: {} -> {:.4f}".format(settings, best), flush=True)
    overall, chosen = break_tie(trials, best)
    if chosen != settings:
        print("tied with the best (within {} of {:.4f}); best over all {} versions: {} -> "
              "{:.4f}".format(TIED_WITHIN, best, len(LEVELS), chosen, overall))
    return chosen


def stage_costs():
    """Stage 3: choose the first model's Viterbi costs on its out-of-fold answers."""
    items = load_checks()
    reference = np.mean([score_levels(i["truths"], i["runs"], i["rules"]) for i in items], axis=0)
    print("possession_v2, same matches and damage: {}".format(np.round(reference, 3).tolist()))
    settings = search_costs(scorer(items), dict(START_COSTS), COST_SEARCH, COST_COLUMNS)
    with open(costs_path(), "w") as handle:
        json.dump(settings, handle)
    print("chosen:", settings)


# Stage 4: the second look -------------------------------------------------------------------------

def second_points(second_booster, points, blocks, first_costs):
    """The second look's log-probabilities for a whole match, half by half, and 0 outside the
    halves, as the first model's."""
    out = np.zeros((len(points), 3), dtype=np.float32)
    for _, first, last in blocks:
        out[first:last + 1] = model.second_look(second_booster, points[first:last + 1], first_costs)
    return out


def second_rows(item, first_costs, offset):
    """(X, y, names): the second look's training rows from one item of load_checks: every
    EVERY_NTH_FRAME-th graded frame from offset, as it is and with the teams swapped."""
    frames, direct, swapped, names = [], [], [], None
    for _, first, last in item["blocks"]:
        half = item["points"][first:last + 1]
        features, names = model.second_features(half, first_costs)
        direct.append(features)
        swapped.append(model.second_features(half[:, model.SWAP], first_costs)[0])
        frames.append(np.arange(first, last + 1))
    frames, direct, swapped = (np.concatenate(part) for part in (frames, direct, swapped))
    truth = item["truths"][0][frames]  # the labels as given
    keep = np.zeros(len(frames), dtype=bool)
    keep[offset::EVERY_NTH_FRAME] = True
    keep &= truth != pm.X
    labels = truth[keep]
    return (np.concatenate([direct[keep], swapped[keep]]),
            np.concatenate([labels, np.where(labels == pm.D, pm.D, 1 - labels)]), names)


def stack(rows, picked, part):
    """One part of rows from second_rows, 0 the features and 1 the labels, for the items picked."""
    return np.concatenate([row[part] for row, pick in zip(rows, picked) if pick])


def stage_second():
    """Stage 4: train the second look without each training match in turn, on the first model's
    out-of-fold answers, and choose its own costs on its answers for the matches left out."""
    import lightgbm
    with open(costs_path()) as handle:
        first_costs = json.load(handle)
    with open(os.path.join(WORK_DIR, "rounds.json")) as handle:
        cv_rounds = json.load(handle)
    first_table = model.cost_table(first_costs["switch"], first_costs["stop"],
                                   first_costs["restart"])
    start = time.time()
    items = load_checks(with_rules=False)
    rows = [second_rows(item, first_table, index % EVERY_NTH_FRAME)
            for index, item in enumerate(items)]
    names = rows[0][2]
    print("{} features for the second look, rows ready in {:.0f} s".format(
        len(names), time.time() - start), flush=True)

    rounds, looks = [], [None] * len(items)
    for left_out in TRAIN:
        start = time.time()
        out = [item["name"] == left_out for item in items]
        kept = [not left for left in out]
        train = lightgbm.Dataset(stack(rows, kept, 0), stack(rows, kept, 1), feature_name=names,
                                 free_raw_data=True)
        valid = lightgbm.Dataset(stack(rows, out, 0), stack(rows, out, 1), reference=train,
                                 free_raw_data=True)
        booster = lightgbm.train(SECOND_LIGHTGBM, train, num_boost_round=MOST_ROUNDS,
                                 valid_sets=[valid],
                                 callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)])
        rounds.append(booster.best_iteration)
        for index, left in enumerate(out):
            if left:
                looks[index] = dict(items[index], points=second_points(
                    booster, items[index]["points"], items[index]["blocks"], first_table))
        print("without {}: {} rows, best round {}, log loss {:.4f} ({:.0f} s)".format(
            left_out, sum(len(row[1]) for row, keep in zip(rows, kept) if keep),
            booster.best_iteration,
            booster.best_score["valid_0"]["multi_logloss"], time.time() - start), flush=True)

    print("\ncosts for the second look:")
    chosen = search_costs(scorer(looks), dict(START_COSTS), COST_SEARCH, COST_COLUMNS)
    with open(second_path(), "w") as handle:
        json.dump({"rounds": rounds, "costs": chosen, "first_costs": first_costs,
                   "cv_rounds": cv_rounds}, handle)
    print("chosen:", chosen)
    show_second(items, first_costs, looks, chosen)


def show_second(items, first_costs, looks, second_costs):
    """The first model and the second look on the matches left out, each at its own costs."""
    print("\n  {:<26} {}  {:>5}   as given, per match".format("", " ".join(
        "{:>6}".format("{:.0f}s".format(l) if l else "given") for l in LEVELS), "poss"))
    means = []
    for title, chosen, scored in (("model", first_costs, items),
                                  ("+ second look", second_costs, looks)):
        costs = model.cost_table(chosen["switch"], chosen["stop"], chosen["restart"])
        rows, counts = [], []
        for item in scored:
            codes = decode(item["points"], item["blocks"], costs)
            rows.append(score_levels(item["truths"], item["runs"], codes))
            counts.append(len(pm.possession_spells(codes, item["runs"])))
        rows = np.array(rows)
        means.append(rows.mean(axis=0))
        per_match = [rows[[item["name"] == name for item in scored], 0].mean() for name in TRAIN]
        print("  {:<26} {}  {:>5.0f}   {}".format(
            "{} ({:g} {:g} {:g})".format(title, chosen["switch"], chosen["stop"],
                                         chosen["restart"]),
            " ".join("{:>6.4f}".format(m) for m in means[-1]), np.mean(counts),
            " ".join("{:.3f}".format(s) for s in per_match)))
    print("  {:<26} {}".format("gain", " ".join("{:>+6.4f}".format(g)
                                                for g in means[1] - means[0])))


# Stage 5: final -----------------------------------------------------------------------------------

def load_second(costs, cv_rounds):
    """What 'second' chose, or None if it has not been run here. If 'cv' or 'costs' ran again after
    it, its rounds and costs no longer fit the first model, so that is an error."""
    flags = " --all" if set(HELD_BACK) <= set(TRAIN) else ""
    if not os.path.isfile(second_path()):
        print("no second look: run 'second{}' before 'final{}' to add one".format(flags, flags))
        return None
    with open(second_path()) as handle:
        second = json.load(handle)
    if second["first_costs"] != costs or second["cv_rounds"] != cv_rounds:
        raise RuntimeError("'cv' or 'costs' ran again after 'second': run 'second{}' "
                           "again".format(flags))
    return second


def final_second(folder, first_costs, second):
    """Train the second look on the first model's out-of-fold answers for every training match and
    save it in folder. Returns the settings that name it."""
    import lightgbm
    first_table = model.cost_table(first_costs["switch"], first_costs["stop"],
                                   first_costs["restart"])
    rows = [second_rows(item, first_table, index % EVERY_NTH_FRAME)
            for index, item in enumerate(load_checks(with_rules=False))]
    every = [True] * len(rows)
    rounds = int(round(np.mean(second["rounds"])))
    start = time.time()
    booster = lightgbm.train(SECOND_LIGHTGBM, lightgbm.Dataset(
        stack(rows, every, 0), stack(rows, every, 1), feature_name=rows[0][2]),
        num_boost_round=rounds)
    path = os.path.join(folder, model.SECOND_FILE)
    booster.save_model(path)
    print("second look: {} rounds on {} rows in {:.0f} s, its costs {:g} {:g} {:g}".format(
        rounds, sum(len(row[1]) for row in rows), time.time() - start,
        second["costs"]["switch"], second["costs"]["stop"], second["costs"]["restart"]))
    print("second look file: {} ({:.1f} MB)".format(path, os.path.getsize(path) / 1e6))
    return {"second_model": model.SECOND_FILE, "second_rounds": rounds,
            "second_costs": second["costs"]}


def stage_final():
    """Stage 5: train the first model on every training match and save it with its settings: into
    models/ with --all, as the model to submit, and otherwise beside the work files, for 'test'.
    The second look is trained and saved beside it when 'second' has been run."""
    import lightgbm
    with open(os.path.join(WORK_DIR, "rounds.json")) as handle:
        cv_rounds = json.load(handle)
    rounds = int(round(np.mean(cv_rounds)))  # the average best number of trees over the folds
    with open(costs_path()) as handle:
        costs = json.load(handle)
    second = load_second(costs, cv_rounds)
    to_submit = set(HELD_BACK) <= set(TRAIN)
    model_path, settings_path = ((model.MODEL_PATH, model.SETTINGS_PATH) if to_submit
                                 else trial_paths())
    X, y, names = load_rows(TRAIN)
    start = time.time()
    train = lightgbm.Dataset(X, y, feature_name=names)
    booster = lightgbm.train(LIGHTGBM, train, num_boost_round=rounds, valid_sets=[train],
                             callbacks=[lightgbm.log_evaluation(LOG_EVERY)])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    booster.save_model(model_path)
    settings = {"switch_cost": costs["switch"], "stop_cost": costs["stop"],
                "restart_cost": costs["restart"], "rounds": rounds, "trained_on": list(TRAIN),
                "held_back": [name for name in HELD_BACK if name not in TRAIN],
                "copies_per_match": COPIES_PER_MATCH, "every_nth_frame": EVERY_NTH_FRAME,
                "feature_names": names}
    print("trained {} rounds on {} rows in {:.0f} s".format(rounds, len(y), time.time() - start))
    print("model file: {} ({:.1f} MB)".format(model_path, os.path.getsize(model_path) / 1e6))
    if second is not None:
        settings.update(final_second(os.path.dirname(settings_path), costs, second))
    with open(settings_path, "w") as handle:
        json.dump(settings, handle, indent=2)
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gains)[:20]
    print("features that mattered most:")
    for index in order:
        print("  {:<20} {:>5.1f}%".format(names[index], 100.0 * gains[index] / gains.sum()))


# Stage 6: test ------------------------------------------------------------------------------------

def run_on(booster, name, seed, damage_settings=None):
    """(match, broken, points) for one held-back match: the match, its damaged copy and the first
    model's log-probabilities. damage_settings changes test_on_other_matches' damage for this copy
    only, for the harder kinds of damage in 'test'."""
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
                                                  speed[first:stop], team[first:stop],
                                                  match["pitch"])
    return match, broken, points


def stage_test():
    """Stage 6: score the trial model on the three held-back matches, under the default damage and
    eight harder kinds, beside possession_v1 and possession_v2. These matches chose nothing, so
    this is the honest score."""
    if set(HELD_BACK) <= set(TRAIN):
        print("nothing is held back with --all: run 'test' before it, without --all")
        return
    model_path, settings_path = trial_paths()
    if not os.path.isfile(model_path):
        print("no model at {}: run 'final' first, without --all".format(model_path))
        return
    booster, settings = model.load_model(model_path, settings_path)
    seen = sorted(set(settings["trained_on"]) & set(HELD_BACK))
    if seen:
        print("this model trained on {}, so it cannot be tested on them".format(", ".join(seen)))
        return
    costs = model.cost_table(settings["switch_cost"], settings["stop_cost"],
                             settings["restart_cost"])
    print("model:", model_path)
    print("costs:", settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    second_booster = model.load_second_model(settings, settings_path)
    if second_booster is not None:
        second_costs = model.cost_table(settings["second_costs"]["switch"],
                                        settings["second_costs"]["stop"],
                                        settings["second_costs"]["restart"])
        print("'+ second look' is the same model read again by the second look ({} rounds), with "
              "its own costs: {:g} {:g} {:g}".format(
                  settings["second_rounds"], settings["second_costs"]["switch"],
                  settings["second_costs"]["stop"], settings["second_costs"]["restart"]))
    scenarios = (("default damage", {}),
                 ("stoppage path started about a second off its true frame (leak check)",
                  {"DEAD_START_SHIFT_S": 1.0}),
                 ("ball lost twice as often", {"LOSS_RATE": 2.0 * other.LOSS_RATE}),
                 ("ball found half as fast", {"FIND_RATE": 0.5 * other.FIND_RATE}),
                 ("short losses left as gaps", {"FILL_S": 0.0}),
                 ("short losses drawn in where the ball really was", {"FILL_TRUE": 1.0}),
                 ("shaky positions, as the first damage had", {"BALL_NOISE_M": 0.4,
                                                               "PLAYER_NOISE_M": 0.25}),
                 ("many wrong-team tracks", {"WRONG_TEAM": 0.20}),
                 ("no ball at all", {"LOSS_RATE": 1e6, "FIND_RATE": 0.0, "FILL_S": 0.0,
                                     "GLITCH_RATE": 0.0}))
    for title, damage_settings in scenarios:
        rows, macro, counts, per_match = {}, {}, {}, {}
        for name in HELD_BACK:
            seed = other.SEED + other.MATCHES.index(STEMS[name])
            match, broken, points = run_on(booster, name, seed, damage_settings)
            codes = {"model": decode(points, match["blocks"], costs)}
            if second_booster is not None:
                codes["+ second look"] = decode(
                    second_points(second_booster, points, match["blocks"], costs),
                    match["blocks"], second_costs)
            codes["possession_v2"] = other.run_method(possession_v2, broken)
            codes["possession_v1"] = other.run_method(possession_v1, broken)
            truth = match["truth"]
            runs = pm.valid_runs(truth)
            truths = [labels_at(truth, level) for level in LEVELS]
            for key, predicted in codes.items():
                scores = score_levels(truths, runs, predicted)
                rows.setdefault(key, []).append(scores)
                per_match.setdefault(key, []).append(scores[0])
                macro.setdefault(key, []).append(
                    pm.macro_f1(pm.scored_confusion(pm.confusion(truth, predicted))))
                counts.setdefault(key, []).append(len(pm.possession_spells(predicted, runs)))
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
    parser.add_argument("stage", choices=("features", "cv", "costs", "second", "final", "test"))
    parser.add_argument("--all", action="store_true",
                        help="all seven matches, for the model to submit (see the docstring)")
    args = parser.parse_args()
    if args.all:
        TRAIN = TUNING + HELD_BACK
        WORK_DIR = WORK_DIR_ALL
    {"features": stage_features, "cv": stage_cv, "costs": stage_costs, "second": stage_second,
     "final": stage_final, "test": stage_test}[args.stage]()
