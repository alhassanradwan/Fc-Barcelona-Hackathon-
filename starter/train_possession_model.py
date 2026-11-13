"""Train the possession model on public matches, choose its Viterbi costs, and test it.

    python starter/train_possession_model.py features   # damaged copies of the tuning matches
    python starter/train_possession_model.py cv         # hold out each tuning match in turn
    python starter/train_possession_model.py costs      # choose the costs on those predictions
    python starter/train_possession_model.py second     # the second look, on the same predictions
    python starter/train_possession_model.py final      # train on all four, kept with the work files
    python starter/train_possession_model.py test       # score that model on the three held-back

    python starter/train_possession_model.py features --all    # then cv, costs, second, final --all

Add --touch 2 to every stage to try labels without brief touches (see below). Each version of the
labels keeps its own work files and models, so the two can be tested side by side.

Add --change to cv, costs, final and test for the change model (see below). It uses the features
and the first model's 'cv' from the runs without it, and keeps its own costs and models:

    python starter/train_possession_model.py cv --change      # the change model, each match left out
    python starter/train_possession_model.py costs --change   # the costs again, with its slope
    python starter/train_possession_model.py final --change
    python starter/train_possession_model.py test --change    # 'model, one cost' is the same model
                                                              # with the costs chosen without it

None of the competition's data is read here. Training uses IDSSE matches only, as described in
test_on_other_matches.py, so the only labels the model has seen are from other games (RULES.md).

The split is by match. Frames from one match are so alike that a random split would report a score
far above what the model can really do. J03WN1, J03WOY and J03WR9 are never used for training or for
choosing anything in this file, except with --all.

--all makes the model to submit, once 'test' has shown the method works: the same stages on all
seven matches, each left out in turn to choose the rounds and costs, with its files kept apart. Only
'final --all' writes to models/, and 'test' never reads from there, so a model that has seen the
held-back matches is never scored on them.

Which labels the model learns
-----------------------------
By default, the labels as given. They are made the way CHALLENGE.md says the Barcelona answers are:
read frame by frame from the tracking's ball owner (BallPossession here), with nothing smoothed.
With --touch S, a spell under S seconds between two of the rival's is handed back to the rival
before training, as in the 'S s' columns of 'test'. At 2 s that relabels about 2% of the live frames
and removes about a quarter of the possessions. Tried with --touch 2, it scored lower on the
held-back matches as given (0.539 against 0.594) and at 2 s (0.646 against 0.660). It scored higher
at 5 and 8 s mainly because it makes fewer possessions: with a higher switch cost, the model trained
on the labels as given comes within about 0.01 of it there on the tuning matches. The costs are
chosen on the same version of the labels the model learns, and never on what the public five
minutes look like (RULES.md).

The change model
----------------
LightGBM again, on the same rows and features, answering only whether a change of team in play, H
to A or A to H, happens within CHANGE_NEAR frames. Viterbi then charges switch_cost for a change
where that chance is its training average, less where it is higher and more where it is lower, as
much as change_slope says. A slope of 0 is the same as having no change model, and 'costs' tries
it, so the costs can always fall back to one switch cost for every frame.

Tried on the held-back matches, it made no difference: 0.594 as given with it and without it, and
up to 0.006 lower on the other versions. It sees the same features as the first model, so it
probably has little to add, and the model to submit does not use it.

The second look
---------------
A second, smaller LightGBM model that reads the first model's answers instead of the tracking: its
three log-probabilities at each frame, its team edge and dead-ball chance up to 2 s either way and
averaged over windows before and after, the lowest and highest edge nearby, and how long its pick
and its Viterbi labels have held and will hold, with what comes before and after. It learns from the
out-of-fold answers 'cv' saved, so it learns the mistakes the first model makes on a match it has
not seen, such as a flicker to the other team or a change drawn late. 'second' trains it without
each training match in turn, as 'cv' does, and chooses its own costs on its answers for the match
left out, by the same search as 'costs'.

With the first model as it was before the clues to restarts below, it gained 0.005 as given and
0.011 to 0.012 at 2, 5 and 8 s on the tuning matches. On the held-back matches under the default
damage, which chose nothing, it gained 0.007 as given, better on each of the three, and 0.008 to
0.011 at 2, 5 and 8 s. 'final' adds it beside the first model when 'second' has been run, and 'test'
scores it as '+ second look'. The first model's own costs stay in switch_cost, stop_cost and
restart_cost, and the second look's are kept apart in second_costs, which possession_model.py uses
whenever the settings name a second look.

Clues to restarts
-----------------
After the ball goes out, the team that restarts play has it, and where the players stand often
shows which team that is even when the ball itself is lost: a throw-in is taken at a touchline, a
corner at a corner flag, a free kick into the box with players waiting in it, and a kick-off with
both teams in their own halves. The first model's last 13 features are for these
(restart_features in possession_model.py).

They were tested before they were added, with the rule set before running, and with the costs and
the second look made again as 'costs' and 'second' make them. On the tuning matches, each left out
in turn, the whole model gained 0.008 as given and 0.006, 0.005 and 0.004 at 2, 5 and 8 s. On the
held-back matches, over the default damage and three harder kinds from 'test' (ball lost twice as
often, found half as fast, many wrong-team tracks), it gained 0.014 as given and 0.009, 0.005 and
0.005 at 2, 5 and 8 s: 0.011 as given under the default damage, and most, 0.024, when the ball is
found slowly.

Why every copy is damaged differently
-------------------------------------
The damage copies what measure_tracking.py measured on the ATD, but that is one match, and some
errors it cannot measure at all, so a model trained on one fixed kind of damage could learn that
damage rather than football. Each training copy draws its own damage from ranges around the measured
settings, so the model cannot lean on any one pattern, such as a missing ball always meaning a
stoppage. The 'test' stage also scores kinds of damage outside the training mix.
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
TRAIN = TUNING  # all seven with --all
STEMS = {stem[-6:]: stem for stem in other.MATCHES}
WORK_DIR = os.path.join(other.DATA_DIR, "model_work")
WORK_DIR_ALL = os.path.join(other.DATA_DIR, "model_work_all")

COPIES_PER_MATCH = 3
EVERY_NTH_FRAME = 5  # neighbouring frames say almost the same thing
TRAINING_SEED = 40
CHECK_SEEDS = (11, 12)  # default damage, for comparing with possession_v2 when choosing costs
N_PROCESSES = int(os.environ.get("N_PROCESSES", 8))

# Drawn afresh for every training copy. (name in test_on_other_matches, lowest, highest) as
# multiples of the measured setting there, then as plain values.
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

LIGHTGBM = {"objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l2": 5.0, "seed": 2027, "deterministic": True, "force_row_wise": True,
            "num_threads": int(os.environ.get("LGBM_THREADS", 16)), "verbosity": -1}
MOST_ROUNDS = 800
PATIENCE = 40
LOG_EVERY = 25

SWITCH_COSTS = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
STOP_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
RESTART_COSTS = (8.0, 16.0, 32.0, 64.0, 128.0)
# Costs closer than this to the best spellF1 on the chosen labels count as tied with it, and the tie
# goes to whichever does best over all of LEVELS. 0.002 is about ten possessions out of the several
# thousand in the check copies, far below what EVALUATION.md calls noise.
TIED_WITHIN = 0.002
LEVELS = (0.0, 2.0, 5.0, 8.0)  # seconds of brief touches not counted, 0 being the labels as given
TOUCH_S = 0.0  # the one of LEVELS the model learns and the costs are chosen on, set with --touch

CHANGE = False  # whether to use the change model, set with --change
CHANGE_NEAR = 12  # frames either side of a change of team that count as near it, about 0.5 s
CHANGE_LIGHTGBM = dict({key: value for key, value in LIGHTGBM.items() if key != "num_class"},
                       objective="binary", metric="binary_logloss")
SLOPES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)  # how far the change model moves the switch cost

SECOND_LIGHTGBM = dict(LIGHTGBM, num_leaves=31)  # fewer features than the first, all of one kind

# Where the costs search starts, what it tries in turn, and how it prints them: (name, width,
# decimals). The change model adds its slope to these.
START_COSTS = {"switch": 24.0, "stop": 32.0, "restart": 32.0}
COST_SEARCH = (("switch", SWITCH_COSTS), ("stop", STOP_COSTS), ("restart", RESTART_COSTS),
               ("switch", SWITCH_COSTS))
COST_COLUMNS = (("switch", 7, 0), ("stop", 6, 0), ("restart", 7, 0))
# After the coarse grid, FINE_COSTS=1 walks each cost by these factors around the best so far,
# repeating while it still improves, so the costs are not stuck on the grid's powers of two.
FINE_COSTS = os.environ.get("FINE_COSTS") == "1"
FINE_FACTORS = (0.6, 0.75, 0.875, 1.15, 1.33, 1.6)
FINE_ROUNDS = 3
# COST_LEVELS="2,5" chooses the costs on the mean spellF1 over those versions of the labels (seconds
# of brief touches handed back) rather than on TOUCH_S alone. CHALLENGE.md says brief touches do
# not count in the Barcelona answers, so these may sit nearer them than the labels as given.
COST_LEVELS = tuple(float(v) for v in os.environ.get("COST_LEVELS", "").split(",") if v)
# WORK_SUFFIX keeps a variant's costs, second look and model apart from the default work files.
WORK_SUFFIX = os.environ.get("WORK_SUFFIX", "")


def as_arrays(match):
    """(ball, xy, speed, team) for the frames of the match."""
    return match["ball"], match["xy"], match["speed"], match["team_of"]


def labels_at(truth, touch_s):
    """truth with spells under touch_s seconds between two of the rival's handed back."""
    return other.absorb_touches(truth, touch_s) if touch_s else truth


def trial_paths():
    """Where 'final' without --all keeps its model, the one 'test' scores."""
    stem = "possession_lgbm_change" if CHANGE else "possession_lgbm"
    return os.path.join(WORK_DIR, stem + ".txt"), os.path.join(WORK_DIR, stem + ".json")


def costs_path():
    """Where 'costs' keeps what it chose, apart for the change model."""
    return os.path.join(WORK_DIR, "costs_change.json" if CHANGE else "costs.json")


def second_path():
    """Where 'second' keeps what it chose."""
    return os.path.join(WORK_DIR, "second.json")


def randomised_damage(match, seed):
    """test_on_other_matches.damage with its settings drawn from DAMAGE_SCALES, DAMAGE_VALUES and
    SITUATION_POWER."""
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
    """One damaged copy of one match as sampled feature rows, saved to disk. Teams both ways round.

    The folder and the labels come with the job, because the worker processes never see what the
    command line changed."""
    name, copy, work_dir, touch_s = job
    path = os.path.join(work_dir, "train_{}_{}.npz".format(name, copy))
    if os.path.isfile(path):
        return path

    match = other.load_match(STEMS[name], other.DATA_DIR)
    seed = TRAINING_SEED + 100 * other.MATCHES.index(STEMS[name]) + copy
    broken = randomised_damage(match, seed)
    ball, xy, speed, team = as_arrays(broken)
    targets = labels_at(match["truth"], touch_s)
    rows, labels = [], []
    for _, first, last in match["blocks"]:
        stop = last + 1
        offset = (seed + first) % EVERY_NTH_FRAME
        truth = targets[first:stop][offset::EVERY_NTH_FRAME]
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
    jobs = [(name, copy, WORK_DIR, TOUCH_S) for name in TRAIN for copy in range(COPIES_PER_MATCH)]
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


def near_change(truth):
    """1 for the frames within CHANGE_NEAR of a change of team in play, H to A or A to H, else 0."""
    in_play = (truth == pm.H) | (truth == pm.A)
    changes = np.flatnonzero(in_play[1:] & in_play[:-1] & (truth[1:] != truth[:-1])) + 1
    marks = np.zeros(len(truth) + 1)
    np.add.at(marks, np.maximum(changes - CHANGE_NEAR, 0), 1.0)
    np.add.at(marks, np.minimum(changes + CHANGE_NEAR + 1, len(truth)), -1.0)
    return (np.cumsum(marks)[:-1] > 0.5).astype(np.float32)


def row_frames(name, copy, blocks):
    """(frame, swapped) behind each row training_copy saved for this copy, in the same order."""
    seed = TRAINING_SEED + 100 * other.MATCHES.index(STEMS[name]) + copy
    frames, swapped = [], []
    for _, first, last in blocks:
        picked = np.arange(first, last + 1)[(seed + first) % EVERY_NTH_FRAME::EVERY_NTH_FRAME]
        for flip in (False, True):
            frames.append(picked)
            swapped.append(np.full(len(picked), flip))
    return np.concatenate(frames), np.concatenate(swapped)


def change_targets(names):
    """For the rows load_rows gives for these matches, 1 near a change of team and 0 elsewhere.

    The rows were saved without these, so their frames are worked out again, and checked against
    the labels saved with them."""
    out = []
    for name in names:
        match = other.load_match(STEMS[name], other.DATA_DIR)
        truth = labels_at(match["truth"], TOUCH_S)
        near = near_change(truth)
        for copy in range(COPIES_PER_MATCH):
            frames, swapped = row_frames(name, copy, match["blocks"])
            labels = truth[frames]
            expected = np.where(swapped & (labels != pm.D), 1 - labels, labels)
            path = os.path.join(WORK_DIR, "train_{}_{}.npz".format(name, copy))
            with np.load(path) as saved:
                if not np.array_equal(expected, saved["y"]):
                    raise RuntimeError("{} does not hold the rows this code expects: delete it and "
                                       "run 'features' again".format(path))
            out.append(near[frames])
    return np.concatenate(out)


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


def second_points(second_booster, points, blocks, first_costs):
    """The second look's log-probabilities for a whole match, half by half, and 0 outside the
    halves, as the first model's."""
    out = np.zeros((len(points), 3), dtype=np.float32)
    for _, first, last in blocks:
        out[first:last + 1] = model.second_look(second_booster, points[first:last + 1], first_costs)
    return out


def second_rows(item, first_costs, offset):
    """(X, y, names): the second look's training rows from one item of load_checks, every
    EVERY_NTH_FRAME graded frame from offset, as it is and with the teams swapped."""
    frames, direct, swapped, names = [], [], [], None
    for _, first, last in item["blocks"]:
        half = item["points"][first:last + 1]
        features, names = model.second_features(half, first_costs)
        direct.append(features)
        swapped.append(model.second_features(half[:, model.SWAP], first_costs)[0])
        frames.append(np.arange(first, last + 1))
    frames, direct, swapped = (np.concatenate(part) for part in (frames, direct, swapped))
    truth = item["truths"][LEVELS.index(TOUCH_S)][frames]
    keep = np.zeros(len(frames), dtype=bool)
    keep[offset::EVERY_NTH_FRAME] = True
    keep &= truth != pm.X
    labels = truth[keep]
    return (np.concatenate([direct[keep], swapped[keep]]),
            np.concatenate([labels, np.where(labels == pm.D, pm.D, 1 - labels)]), names)


def stack(rows, picked, part):
    """One part of rows from second_rows, 0 the features and 1 the labels, for the items picked."""
    return np.concatenate([row[part] for row, pick in zip(rows, picked) if pick])


def check_change(change_booster, name, seed):
    """The change model's chance for the whole of one match under the default damage."""
    match = other.load_match(STEMS[name], other.DATA_DIR)
    ball, xy, speed, team = as_arrays(other.damage(match, seed))
    chance = np.zeros(len(ball), dtype=np.float32)
    for _, first, last in match["blocks"]:
        stop = last + 1
        direct, swapped = model.team_features(ball[first:stop], xy[first:stop], speed[first:stop],
                                              team[first:stop], match["pitch"])
        chance[first:stop] = model.change_from(change_booster, direct, swapped)
    return chance


def stage_cv_change():
    """The change model without each training match in turn, keeping its chances for the one left
    out. The first model's predictions come from 'cv' without --change."""
    import lightgbm
    results = []
    for left_out in TRAIN:
        start = time.time()
        train_names = [name for name in TRAIN if name != left_out]
        X, _, names = load_rows(train_names)
        X_valid, _, _ = load_rows([left_out])
        near, near_valid = change_targets(train_names), change_targets([left_out])
        train = lightgbm.Dataset(X, near, feature_name=names, free_raw_data=True)
        valid = lightgbm.Dataset(X_valid, near_valid, reference=train, free_raw_data=True)
        booster = lightgbm.train(CHANGE_LIGHTGBM, train, num_boost_round=MOST_ROUNDS,
                                 valid_sets=[valid],
                                 callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False),
                                            lightgbm.log_evaluation(LOG_EVERY)])
        results.append(booster.best_iteration)
        print("without {}: {} rows, {:.1%} near a change, best round {}, log loss {:.4f} "
              "({:.0f} s)".format(left_out, len(near), float(near.mean()), booster.best_iteration,
                                  booster.best_score["valid_0"]["binary_logloss"],
                                  time.time() - start), flush=True)
        for seed in CHECK_SEEDS:
            np.save(os.path.join(WORK_DIR, "oofc_{}_{}.npy".format(left_out, seed)),
                    check_change(booster, left_out, seed))
    with open(os.path.join(WORK_DIR, "rounds_change.json"), "w") as handle:
        json.dump(results, handle)
    print("best rounds:", results)


def stage_cv():
    """Train without each training match in turn, and keep the predictions for the one left out."""
    if CHANGE:
        return stage_cv_change()
    import lightgbm
    results = []
    for left_out in TRAIN:
        start = time.time()
        train_names = [name for name in TRAIN if name != left_out]
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


def load_checks(with_rules=True):
    """The training matches under default damage, with their out-of-fold points and, with_rules,
    what possession_v2 makes of them."""
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
            if CHANGE:
                item["chance"] = np.load(os.path.join(WORK_DIR,
                                                      "oofc_{}_{}.npy".format(name, seed)))
            items.append(item)
    return items


def decode(points, blocks, costs, switches=None):
    """Viterbi labels for every half, with switches as the cost of a change of team at each frame
    if it is given."""
    codes = np.full(len(points), pm.X, dtype=np.int8)
    for _, first, last in blocks:
        stop = last + 1
        codes[first:stop] = model.best_labels(points[first:stop].astype(np.float64), costs,
                                              None if switches is None else switches[first:stop])
    return codes


def break_tie(trials, best):
    """(overall, settings) of the trial within TIED_WITHIN of best that does best over all LEVELS.

    trials holds (settings, score on the chosen labels, mean score over all of LEVELS)."""
    tied = [(overall, settings) for settings, score, overall in trials
            if score >= best - TIED_WITHIN]
    return max(tied, key=lambda pair: pair[0])


def scorer(items, rate=None):
    """evaluate(trial) for search_costs: (spellF1 on the chosen labels, on each of LEVELS, and the
    possessions per match) for the items' points decoded with the trial's costs, and with its slope
    when it has one, rate being the change model's training average."""
    choice = LEVELS.index(TOUCH_S)

    def evaluate(trial):
        costs = model.cost_table(trial["switch"], trial["stop"], trial["restart"])
        rows, counts = [], []
        for item in items:
            switches = None
            if "slope" in trial:
                switches = model.switch_by_frame(item["chance"], {
                    "switch_cost": trial["switch"], "change_slope": trial["slope"],
                    "change_rate": rate})
            codes = decode(item["points"], item["blocks"], costs, switches)
            rows.append(score_levels(item["truths"], item["runs"], codes))
            counts.append(len(pm.possession_spells(codes, item["runs"])))
        means = np.mean(rows, axis=0)
        if COST_LEVELS:  # choose on the mean over these versions of the labels instead
            target = float(np.mean([means[LEVELS.index(level)] for level in COST_LEVELS]))
            return target, means, float(np.mean(counts))
        return float(means[choice]), means, float(np.mean(counts))
    return evaluate


def search_costs(evaluate, settings, search, columns):
    """The settings that do best on the chosen labels, trying each of search's (name, values) in
    turn from settings, near-ties settled by break_tie. Prints every trial."""
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
    if FINE_COSTS:
        tried = {tuple(sorted(t.items())) for t, _, _ in trials}
        for round_index in range(FINE_ROUNDS):
            improved = False
            for key in ("switch", "stop", "restart"):
                for factor in FINE_FACTORS:
                    trial = dict(settings, **{key: round(settings[key] * factor, 2)})
                    signature = tuple(sorted(trial.items()))
                    if signature in tried:
                        continue
                    tried.add(signature)
                    score, means, count = evaluate(trial)
                    print("{} | {}  {:>6.4f} {:>5.0f}  fine".format(
                        " ".join("{:>{}.{}f}".format(trial[column], width, max(places, 2))
                                 for column, width, places in columns),
                        " ".join("{:>6.3f}".format(m) for m in means), score, count), flush=True)
                    trials.append((trial, score, float(np.mean(means))))
                    if score > best:
                        best, settings, improved = score, trial, True
            print("  fine round {}: {} -> {:.4f}".format(round_index + 1, settings, best),
                  flush=True)
            if not improved:
                break
    overall, chosen = break_tie(trials, best)
    if chosen != settings:
        print("tied with the best (within {} of {:.4f}); best over all {} versions: {} -> "
              "{:.4f}".format(TIED_WITHIN, best, len(LEVELS), chosen, overall))
    return chosen


def stage_costs():
    """Choose the Viterbi costs on predictions for matches the model did not train on."""
    if CHANGE and not os.path.isfile(os.path.join(WORK_DIR, "rounds_change.json")):
        print("no change model predictions in {}: run 'cv --change' first".format(WORK_DIR))
        return
    items = load_checks()
    rate = float(np.mean(change_targets(TRAIN))) if CHANGE else None
    reference = np.mean([score_levels(i["truths"], i["runs"], i["rules"]) for i in items], axis=0)
    print("possession_v2, same matches and damage: {}".format(np.round(reference, 3).tolist()))

    settings, search, columns = dict(START_COSTS), list(COST_SEARCH), list(COST_COLUMNS)
    if CHANGE:
        print("change model: {:.1%} of the training rows are near a change of team".format(rate))
        settings["slope"] = 0.0
        search = [("switch", SWITCH_COSTS), ("slope", SLOPES), ("stop", STOP_COSTS),
                  ("restart", RESTART_COSTS), ("switch", SWITCH_COSTS), ("slope", SLOPES)]
        columns.insert(1, ("slope", 6, 2))
    settings = search_costs(scorer(items, rate), settings, search, columns)
    with open(costs_path(), "w") as handle:
        json.dump(settings, handle)
    print("chosen:", settings)


def stage_second():
    """Train the second look without each training match in turn, on the first model's out-of-fold
    answers, and choose its costs on its answers for the matches left out."""
    import lightgbm
    if CHANGE:
        print("the second look reads the first model alone: run 'second' without --change")
        return
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

    for look in looks:  # out-of-fold second-look points, for tuning anything after Viterbi
        np.save(os.path.join(WORK_DIR, "oof2_{}_{}.npy".format(look["name"], look["seed"])),
                look["points"])

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


def final_change(X, names, folder, slope):
    """Train the change model on the rows the first model trained on, and save it in folder.
    Returns the settings that tell possession_model.py to use it."""
    import lightgbm
    with open(os.path.join(WORK_DIR, "rounds_change.json")) as handle:
        rounds = int(round(np.mean(json.load(handle))))
    near = change_targets(TRAIN)
    start = time.time()
    change_booster = lightgbm.train(CHANGE_LIGHTGBM, lightgbm.Dataset(X, near, feature_name=names),
                                    num_boost_round=rounds)
    path = os.path.join(folder, model.CHANGE_FILE)
    change_booster.save_model(path)
    print("change model: {} rounds in {:.0f} s, {:.1%} of rows near a change".format(
        rounds, time.time() - start, float(near.mean())))
    print("change model file: {} ({:.1f} MB)".format(path, os.path.getsize(path) / 1e6))
    return {"change_model": model.CHANGE_FILE, "change_slope": slope,
            "change_rate": float(near.mean()), "change_near": CHANGE_NEAR, "change_rounds": rounds}


def load_second(costs, cv_rounds):
    """What 'second' chose, or None if it has not been run here. If 'cv' or 'costs' ran again after
    it, its rounds and costs no longer fit the first model, so that is an error."""
    flags = (" --all" if set(HELD_BACK) <= set(TRAIN) else "") + (
        " --touch {:g}".format(TOUCH_S) if TOUCH_S else "")
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
    """Train on every training match and save the model and its settings: to models/ with --all, as
    the model to submit, and otherwise beside the work files, for 'test'. The second look is trained
    and saved beside them when 'second' has been run."""
    import lightgbm
    with open(os.path.join(WORK_DIR, "rounds.json")) as handle:
        cv_rounds = json.load(handle)
    rounds = int(round(np.mean(cv_rounds)))
    with open(costs_path()) as handle:
        costs = json.load(handle)
    second = None if CHANGE else load_second(costs, cv_rounds)
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
                "restart_cost": costs["restart"], "rounds": rounds, "touch_s": TOUCH_S,
                "trained_on": list(TRAIN),
                "held_back": [name for name in HELD_BACK if name not in TRAIN],
                "copies_per_match": COPIES_PER_MATCH, "every_nth_frame": EVERY_NTH_FRAME,
                "feature_names": names}
    print("trained {} rounds on {} rows in {:.0f} s".format(rounds, len(y), time.time() - start))
    print("model file: {} ({:.1f} MB)".format(model_path, os.path.getsize(model_path) / 1e6))
    if CHANGE:
        settings.update(final_change(X, names, os.path.dirname(settings_path), costs["slope"]))
    elif second is not None:
        settings.update(final_second(os.path.dirname(settings_path), costs, second))
    with open(settings_path, "w") as handle:
        json.dump(settings, handle, indent=2)
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gains)[:20]
    print("features that mattered most:")
    for index in order:
        print("  {:<20} {:>5.1f}%".format(names[index], 100.0 * gains[index] / gains.sum()))


def run_on(booster, change_booster, name, seed, damage_settings=None):
    """(match, broken, points, chance) for one held-back match, chance being None without a change
    model."""
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
    chance = None if change_booster is None else np.zeros(len(ball))
    for _, first, last in match["blocks"]:
        stop = last + 1
        direct, swapped = model.team_features(ball[first:stop], xy[first:stop], speed[first:stop],
                                              team[first:stop], match["pitch"])
        points[first:stop] = model.points_from(booster, direct, swapped)
        if chance is not None:
            chance[first:stop] = model.change_from(change_booster, direct, swapped)
    return match, broken, points, chance


def stage_test():
    """Score on the held-back matches: the default damage, then damage outside the training mix."""
    if set(HELD_BACK) <= set(TRAIN):
        print("nothing is held back with --all: run 'test' before it, without --all")
        return
    model_path, settings_path = trial_paths()
    if not os.path.isfile(model_path):
        print("no model at {}: run 'final{}' first, without --all".format(
            model_path, " --change" if CHANGE else ""))
        return
    booster, settings = model.load_model(model_path, settings_path)
    change_booster = model.load_change_model(settings, settings_path)
    seen = sorted(set(settings["trained_on"]) & set(HELD_BACK))
    if seen:
        print("this model trained on {}, so it cannot be tested on them".format(", ".join(seen)))
        return
    costs = model.cost_table(settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    print("model:", model_path)
    print("costs:", settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    one_cost = None
    if change_booster is not None:
        print("change model: slope {}, {:.1%} of training rows near a change".format(
            settings["change_slope"], settings["change_rate"]))
        if os.path.isfile(os.path.join(WORK_DIR, "costs.json")):
            with open(os.path.join(WORK_DIR, "costs.json")) as handle:
                plain = json.load(handle)
            one_cost = model.cost_table(plain["switch"], plain["stop"], plain["restart"])
            print("'model, one cost' is the same model without it, with the costs 'costs' chose:",
                  plain["switch"], plain["stop"], plain["restart"])
    second_booster = model.load_second_model(settings, settings_path)
    if second_booster is not None:
        second_costs = model.cost_table(settings["second_costs"]["switch"],
                                        settings["second_costs"]["stop"],
                                        settings["second_costs"]["restart"])
        print("'+ second look' is the same model read again by the second look ({} rounds), with "
              "its own costs: {:g} {:g} {:g}".format(
                  settings["second_rounds"], settings["second_costs"]["switch"],
                  settings["second_costs"]["stop"], settings["second_costs"]["restart"]))
    print("labels learned:", "spells under {:.0f} s handed back".format(TOUCH_S) if TOUCH_S
          else "as given")
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
            match, broken, points, chance = run_on(booster, change_booster, name, seed,
                                                   damage_settings)
            switches = None if chance is None else model.switch_by_frame(chance, settings)
            codes = {"model": decode(points, match["blocks"], costs, switches)}
            if one_cost is not None:
                codes["model, one cost"] = decode(points, match["blocks"], one_cost)
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
    parser.add_argument("--touch", type=float, default=0.0, choices=LEVELS,
                        help="learn labels with spells under this many seconds between two of the "
                             "rival's handed back (see the docstring)")
    parser.add_argument("--change", action="store_true",
                        help="add the change model, so a change of team costs less where one is "
                             "likely (cv, costs, final and test; see the docstring)")
    args = parser.parse_args()
    if args.all:
        TRAIN = TUNING + HELD_BACK
        WORK_DIR = WORK_DIR_ALL
    CHANGE = args.change
    TOUCH_S = args.touch
    if TOUCH_S:
        WORK_DIR += "_touch{:.0f}".format(TOUCH_S)
    WORK_DIR += WORK_SUFFIX
    {"features": stage_features, "cv": stage_cv, "costs": stage_costs, "second": stage_second,
     "final": stage_final, "test": stage_test}[args.stage]()
