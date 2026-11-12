"""Which possessions the model misses, and which it invents, on the three held-back matches.

    python starter/check_misses.py

It runs the model that 'final' saved without --all, which never trained on these matches, under the
default damage, exactly as 'test' does. Then it sorts every real possession by simple facts about
it: how long it lasted, how much of it the ball was missing from the tracking, how it started and
ended, and where it began. A real possession counts as found when one of the model's, of the same
team, lines up with it at an IoU of at least 0.5, the middle of spellF1's thresholds, matched one to
one as the scorer does. The model's possessions that match none were invented.

It changes nothing, and reads only the public IDSSE matches (see test_on_other_matches.py).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    sys.path.insert(0, folder)
import possession_metrics as pm  # noqa: E402
import possession_model as model  # noqa: E402
import test_on_other_matches as other  # noqa: E402
import train_possession_model as train  # noqa: E402

# Params
IOU = pm.SPELL_IOU_HEADLINE
DURATIONS = ((1.0, "under 1 s"), (2.0, "1 to 2 s"), (5.0, "2 to 5 s"), (10.0, "5 to 10 s"),
             (np.inf, "10 s or more"))  # (up to, name)
BALL_SEEN = ((0.4, "under 40% of it"), (0.8, "40 to 80% of it"), (np.inf, "80% or more of it"))
STARTS = ("after a stoppage", "won from the other team in play", "at the start of a half")
ENDS = ("by a stoppage", "lost to the other team in play", "at the end of a half")
THIRDS = ("in its own third", "in the middle third", "in its attacking third")
INSTEAD = ("gave it to the other team", "called it dead ball",
           "right team, but joined to a longer possession", "right team, but cut short or split")
INSTEAD_SHORT = ("other team", "dead ball", "joined", "cut short")  # the same, as column names
HOW = ("won in play, then lost in play", "won in play, then a stoppage",
       "after a stoppage, then lost in play", "after a stoppage, then a stoppage")
REALLY = ("the other team had the ball", "the ball was dead", "same team, but the edges were off")


def bucket(value, limits):
    """The name of the first of limits that value is under."""
    for limit, name in limits:
        if value < limit:
            return name
    return limits[-1][1]


def match_both(overlaps, n_real, n_made):
    """(found, matched): the scorer's one-to-one greedy match at IOU, flagged on both sides."""
    found = np.zeros(n_real, dtype=bool)
    matched = np.zeros(n_made, dtype=bool)
    for iou, real_index, made_index in overlaps:
        if iou < IOU:
            break
        if found[real_index] or matched[made_index]:
            continue
        found[real_index] = matched[made_index] = True
    return found, matched


def attacks_up(match):
    """{(first frame of a half, team): whether that team attacks towards larger x in it}, from where
    its players stand on average, as possession_model.face_the_same_way works it out."""
    out = {}
    x = match["xy"][..., 0]
    for _, first, last in match["blocks"]:
        part, team = x[first:last + 1], match["team_of"][first:last + 1]
        home, away = (np.nanmean(np.where(team == side, part, np.nan)) for side in (pm.H, pm.A))
        out[first, pm.H] = bool(home < away)
        out[first, pm.A] = not out[first, pm.H]
    return out


def most_of(codes):
    """The commonest of H, A and D in codes."""
    return int(np.argmax(np.bincount(codes, minlength=pm.N_LABELS)[:pm.D + 1]))


def describe_match(name, booster, costs):
    """(real, made): facts about each real possession and each one the model made, in one match."""
    seed = other.SEED + other.MATCHES.index(train.STEMS[name])
    match, broken, points, _ = train.run_on(booster, None, name, seed)
    codes = train.decode(points, match["blocks"], costs)
    truth = match["truth"]
    runs = pm.valid_runs(truth)
    real_spells = pm.possession_spells(truth, runs)
    made_spells = pm.possession_spells(codes, runs)
    found, matched = match_both(pm.spell_overlaps(real_spells, made_spells), len(real_spells),
                                len(made_spells))
    print("{}: spellF1 as given {:.3f}, {} real possessions, {} found, the model made {}, {} of "
          "them invented".format(name, pm.spell_scores(truth, codes, runs)["mean_f1"],
                                 len(real_spells), int(found.sum()), len(made_spells),
                                 int((~matched).sum())), flush=True)

    ball_seen = ~np.isnan(broken["ball"][:, 0])
    length = match["pitch"][0]
    up = attacks_up(match)
    real = []
    for (start, stop, team), hit in zip(real_spells, found):
        run_start, run_stop = next((a, b) for a, b in runs if a <= start < b)
        half = next(first for _, first, last in match["blocks"] if first <= start <= last)
        along = match["ball"][start, 0] / length
        along = along if up[half, team] else 1.0 - along
        facts = {"found": bool(hit),
                 "duration": bucket((stop - start) / float(pm.FPS), DURATIONS),
                 "ball": bucket(ball_seen[start:stop].mean(), BALL_SEEN),
                 "start": (STARTS[2] if start == run_start
                           else STARTS[0] if truth[start - 1] == pm.D else STARTS[1]),
                 "end": (ENDS[2] if stop == run_stop
                         else ENDS[0] if truth[stop] == pm.D else ENDS[1]),
                 "where": THIRDS[min(2, max(0, int(along * 3)))]}
        facts["how"] = "{}, then {}".format(
            "won in play" if facts["start"] == STARTS[1] else "after a stoppage",
            "lost in play" if facts["end"] == ENDS[1] else "a stoppage")
        if not hit:
            said = most_of(codes[start:stop])
            if said == pm.D:
                facts["instead"] = INSTEAD[1]
            elif said != team:
                facts["instead"] = INSTEAD[0]
            else:
                longest = max((min(stop, b) - max(start, a), b - a) for a, b, k in made_spells
                              if k == team and a < stop and start < b)
                facts["instead"] = INSTEAD[2] if longest[1] > stop - start else INSTEAD[3]
        real.append(facts)

    made = []
    for (start, stop, team), hit in zip(made_spells, matched):
        was = most_of(truth[start:stop])
        made.append({"real": bool(hit),
                     "duration": bucket((stop - start) / float(pm.FPS), DURATIONS),
                     "ball": bucket(ball_seen[start:stop].mean(), BALL_SEEN),
                     "really": REALLY[1] if was == pm.D else REALLY[0] if was != team else REALLY[2]})
    return real, made


def show(title, rows, key, groups, ok, ok_name, bad_name):
    """One table: for each group, how many rows, the share that are ok, and its share of the bad."""
    total_bad = sum(not row[ok] for row in rows)
    print("\n{}".format(title))
    print("  {:<46} {:>6} {:>7} {:>7} {:>8}".format("", "count", ok_name, bad_name, "of all"))
    for group in groups:
        inside = [row for row in rows if row[key] == group]
        if inside:
            bad = sum(not row[ok] for row in inside)
            print("  {:<46} {:>6} {:>6.0f}% {:>7} {:>7.0f}%".format(
                group, len(inside), 100.0 * (len(inside) - bad) / len(inside), bad,
                100.0 * bad / max(total_bad, 1)))


def show_shares(title, rows, key, groups):
    """One table: how many rows fall in each group, and what share of all of them."""
    print("\n{}".format(title))
    for group in groups:
        count = sum(row[key] == group for row in rows)
        print("  {:<46} {:>6} {:>7.0f}%".format(group, count, 100.0 * count / max(len(rows), 1)))


def show_cross(title, rows, row_key, row_groups, column_key, column_groups, column_names):
    """One table of counts: row_key down the side, column_key across the top."""
    print("\n{}".format(title))
    print("  {:<16}".format("") + "".join("{:>12}".format(name) for name in column_names))
    for group in row_groups:
        inside = [row for row in rows if row[row_key] == group]
        print("  {:<16}".format(group) + "".join(
            "{:>12}".format(sum(row[column_key] == column for row in inside))
            for column in column_groups))


def main():
    model_path, settings_path = train.trial_paths()
    if not os.path.isfile(model_path):
        print("no model at {}: run 'train_possession_model.py final' first, without --all".format(
            model_path))
        return
    booster, settings = model.load_model(model_path, settings_path)
    seen = sorted(set(settings["trained_on"]) & set(train.HELD_BACK))
    if seen:
        print("this model trained on {}, so it cannot be checked on them".format(", ".join(seen)))
        return
    costs = model.cost_table(settings["switch_cost"], settings["stop_cost"], settings["restart_cost"])
    print("model: {}, trained on {}, costs {:g} {:g} {:g}".format(
        model_path, ", ".join(settings["trained_on"]), settings["switch_cost"],
        settings["stop_cost"], settings["restart_cost"]))

    real, made = [], []
    for name in train.HELD_BACK:
        more_real, more_made = describe_match(name, booster, costs)
        real += more_real
        made += more_made

    missed = [row for row in real if not row["found"]]
    invented = [row for row in made if not row["real"]]
    print("\nAll three: {} real possessions, {} found ({:.0f}%), {} missed. The model made {}, "
          "{} of them invented ({:.0f}%).".format(
              len(real), len(real) - len(missed), 100.0 * (len(real) - len(missed)) / len(real),
              len(missed), len(made), len(invented), 100.0 * len(invented) / len(made)))

    show("Real possessions, by how long they lasted", real, "duration",
         [name for _, name in DURATIONS], "found", "found", "missed")
    show("Real possessions, by how much of them the ball was in the tracking", real, "ball",
         [name for _, name in BALL_SEEN], "found", "found", "missed")
    show("Real possessions, by how they started", real, "start", STARTS, "found", "found", "missed")
    show("Real possessions, by how they ended", real, "end", ENDS, "found", "found", "missed")
    show("Real possessions, by where they started, as the team in possession sees the pitch", real,
         "where", THIRDS, "found", "found", "missed")
    show_shares("The missed ones: what the model said instead", missed, "instead", INSTEAD)
    show_cross("The missed ones, by how long they lasted and what the model said instead", missed,
               "duration", [name for _, name in DURATIONS], "instead", INSTEAD, INSTEAD_SHORT)
    short = [row for row in real if row["duration"] in (DURATIONS[0][1], DURATIONS[1][1])]
    show("Real possessions under 2 s, by how they started and ended", short, "how", HOW, "found",
         "found", "missed")
    show("The model's possessions, by how long they lasted", made, "duration",
         [name for _, name in DURATIONS], "real", "real", "invent")
    show_shares("The invented ones: what was really happening", invented, "really", REALLY)


if __name__ == '__main__':
    main()
