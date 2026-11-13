"""Clean the Viterbi labels into whole possessions, the objects spellF1 counts.

Two rules, applied inside each half:

  1. A dead-ball run shorter than DEAD_GAP frames with the same team on both sides is handed to
     that team. A brief false stoppage otherwise splits one possession into two, and spellF1 then
     loses the true one (neither half overlaps it enough) and gains a false one.
  2. A team run shorter than MIN_SPELL frames is handed to its neighbours: to the label on both
     sides when they agree, otherwise to the longer neighbour. Each such fragment is a false
     positive and also shrinks the overlap of the possession next to it.

Both lengths are chosen on the tuning matches' out-of-fold second-look answers (saved by
'train_possession_model.py second'), never on the held-back matches and never on the public five
minutes (RULES.md). 'test' then scores the chosen lengths once on the held-back matches.

    python starter/postprocess.py tune        # choose the lengths on the tuning matches
    python starter/postprocess.py test        # score them on the held-back matches
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    sys.path.insert(0, folder)
import possession_metrics as pm  # noqa: E402

DEAD_GAPS = (0, 12, 25, 50, 75, 100)  # frames
MIN_SPELLS = (0, 6, 12, 25, 38, 50, 75)  # frames
SETTINGS_FILE = "postprocess.json"


def runs_of(codes):
    """(starts, stops, labels) of the constant runs of codes."""
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(codes)]))
    return starts, stops, codes[starts]


def fill_short_dead(codes, dead_gap):
    """Rule 1 on one half, in place."""
    if dead_gap <= 0:
        return codes
    starts, stops, labels = runs_of(codes)
    for i in range(1, len(starts) - 1):
        if (labels[i] == pm.D and stops[i] - starts[i] < dead_gap
                and labels[i - 1] == labels[i + 1] and labels[i - 1] in pm.POSSESSION_LABELS):
            codes[starts[i]:stops[i]] = labels[i - 1]
    return codes


def absorb_short_spells(codes, min_spell):
    """Rule 2 on one half, in place, shortest run first, until no short team run is left."""
    if min_spell <= 0:
        return codes
    while True:
        starts, stops, labels = runs_of(codes)
        lengths = stops - starts
        short = [i for i in np.argsort(lengths, kind="stable")
                 if lengths[i] < min_spell and labels[i] in pm.POSSESSION_LABELS
                 and len(starts) > 1]
        if not short:
            return codes
        i = short[0]
        left = labels[i - 1] if i > 0 else None
        right = labels[i + 1] if i < len(starts) - 1 else None
        if left is None or (right is not None and left == right):
            fill = right
        elif right is None:
            fill = left
        else:
            fill = left if lengths[i - 1] >= lengths[i + 1] else right
        codes[starts[i]:stops[i]] = fill


def clean(codes, blocks, dead_gap, min_spell):
    """Both rules on every half of a whole-match label array. Returns a new array."""
    out = codes.copy()
    for _, first, last in blocks:
        half = out[first:last + 1]
        fill_short_dead(half, dead_gap)
        absorb_short_spells(half, min_spell)
        fill_short_dead(half, dead_gap)
    return out


def grid_scores(items, decoded, levels, score_levels):
    """{(dead_gap, min_spell): (mean spellF1 per level, mean possessions)}."""
    out = {}
    for dead_gap in DEAD_GAPS:
        for min_spell in MIN_SPELLS:
            rows, counts = [], []
            for item, codes in zip(items, decoded):
                cleaned = clean(codes, item["blocks"], dead_gap, min_spell)
                rows.append(score_levels(item["truths"], item["runs"], cleaned))
                counts.append(len(pm.possession_spells(cleaned, item["runs"])))
            out[(dead_gap, min_spell)] = (np.mean(rows, axis=0), float(np.mean(counts)))
    return out


def print_grid(scores, levels):
    print("  dead  spell | " + " ".join("{:>6}".format("{:.0f}s".format(l) if l else "given")
                                        for l in levels) + "   mean   poss")
    for (dead_gap, min_spell), (means, count) in scores.items():
        print("  {:>4} {:>6} | {}  {:.4f} {:>6.0f}".format(
            dead_gap, min_spell, " ".join("{:>6.3f}".format(m) for m in means),
            float(np.mean(means)), count), flush=True)


def stage_tune(train):
    with open(train.second_path()) as handle:
        second = json.load(handle)
    costs = train.model.cost_table(second["costs"]["switch"], second["costs"]["stop"],
                                   second["costs"]["restart"])
    items = train.load_checks(with_rules=False)
    for item in items:
        path = os.path.join(train.WORK_DIR, "oof2_{}_{}.npy".format(item["name"], item["seed"]))
        item["points"] = np.load(path)
    decoded = [train.decode(item["points"], item["blocks"], costs) for item in items]
    scores = grid_scores(items, decoded, train.LEVELS, train.score_levels)
    print_grid(scores, train.LEVELS)

    choice = train.LEVELS.index(train.TOUCH_S)
    best = max(means[choice] for means, _ in scores.values())
    # Near-ties go to the one that does best over every version of the truth, as for the costs.
    tied = [(float(np.mean(means)), key) for key, (means, _) in scores.items()
            if means[choice] >= best - train.TIED_WITHIN]
    _, (dead_gap, min_spell) = max(tied)
    base = scores[(0, 0)][0]
    print("\nchosen: dead_gap {} min_spell {}  ->  {} (without: {})".format(
        dead_gap, min_spell, np.round(scores[(dead_gap, min_spell)][0], 4).tolist(),
        np.round(base, 4).tolist()))
    with open(os.path.join(train.WORK_DIR, SETTINGS_FILE), "w") as handle:
        json.dump({"dead_gap": dead_gap, "min_spell": min_spell}, handle)


def stage_test(train):
    with open(os.path.join(train.WORK_DIR, SETTINGS_FILE)) as handle:
        chosen = json.load(handle)
    model_path, settings_path = train.trial_paths()
    booster, settings = train.model.load_model(model_path, settings_path)
    second_booster = train.model.load_second_model(settings, settings_path)
    first = train.model.cost_table(settings["switch_cost"], settings["stop_cost"],
                                   settings["restart_cost"])
    sc = settings["second_costs"]
    second_costs = train.model.cost_table(sc["switch"], sc["stop"], sc["restart"])
    print("held-back matches, default damage; cleanup", chosen)
    rows = {"+ second look": [], "+ cleanup": []}
    counts = {key: [] for key in rows}
    truth_counts = []
    for name in train.HELD_BACK:
        seed = train.other.SEED + train.other.MATCHES.index(train.STEMS[name])
        match, _, points, _ = train.run_on(booster, None, name, seed)
        looked = train.second_points(second_booster, points, match["blocks"], first)
        codes = train.decode(looked, match["blocks"], second_costs)
        cleaned = clean(codes, match["blocks"], chosen["dead_gap"], chosen["min_spell"])
        runs = pm.valid_runs(match["truth"])
        truths = [train.labels_at(match["truth"], level) for level in train.LEVELS]
        truth_counts.append(len(pm.possession_spells(match["truth"], runs)))
        for key, predicted in (("+ second look", codes), ("+ cleanup", cleaned)):
            rows[key].append(train.score_levels(truths, runs, predicted))
            counts[key].append(len(pm.possession_spells(predicted, runs)))
    print("  {:<15} {}  {:>5}   as given, per match".format(
        "", " ".join("{:>6}".format("{:.0f}s".format(l) if l else "given") for l in train.LEVELS),
        "poss"))
    for key in rows:
        print("  {:<15} {}  {:>5.0f}   {}".format(
            key, " ".join("{:>6.3f}".format(m) for m in np.mean(rows[key], axis=0)),
            np.mean(counts[key]), " ".join("{:.3f}".format(r[0]) for r in rows[key])))
    print("  true possessions per match (as given): {:.0f}".format(np.mean(truth_counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("tune", "test"))
    args = parser.parse_args()
    import train_possession_model as train
    {"tune": stage_tune, "test": stage_test}[args.stage](train)
