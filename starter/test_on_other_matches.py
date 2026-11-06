"""Test the possession scripts on other full matches, where the answers are public.

The match this competition is about has five public minutes of ground truth, and RULES.md says they
are not for tuning. Other games are allowed in any amount, so this is the way to measure a method
over whole matches and to choose between versions of it.

    python starter/test_on_other_matches.py --matches J03WN1 J03WOY J03WR9    # the held-back three
    python starter/test_on_other_matches.py                                   # all seven matches

possession_v2 was tuned on J03WMX, J03WOH, J03WPY and J03WQQ, so only the other three say anything
honest about it, and only until they are used to choose between versions too.

The data
--------
IDSSE: seven complete Bundesliga matches from 2022/23 at 25 fps, licensed CC-BY 4.0, data owner DFL.
Bassek et al., Scientific Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y

Every frame of the ball says whether it is in play (BallStatus) and which team has it
(BallPossession), which is everything a frame,label ground truth needs. Download the files into
--data-dir, which defaults to a folder beside this repository so nothing large can be committed:

    https://huggingface.co/datasets/pysport/idsse-data

Why the tracking gets damaged first
-----------------------------------
This tracking is clean and the competition's is not. Scored as it comes, any method would look far
better than it will on the real thing, so each match is also scored after making it look like the
ATD. The published parts of that are from DATA.md: the ball is missing on 41% of graded frames,
disproportionately in crowds and in the air, and about 20.5 of 22 players are tracked at a time.

Two things have to go whatever else is done, because they give the answer away. These files freeze
the ball on every frame of a stoppage, so damage moves it the way a ball really moves then. And the
ball must not vanish or reappear exactly where the true labels change, so the stretches it is hidden
over are deliberately out of line with them.

Everything else under Damage is an assumption, not a measurement. The check on those assumptions is
the baselines: their full-match scores on the real ATD are published in EVALUATION.md and printed
next to their scores here. The closer they agree, the more this test can be trusted to say anything
about the real match. A different league, tracking system and definition of possession mean it is
evidence and never a prediction.

Several versions of the truth
-----------------------------
These labels change team far more often than the public five minutes do, and they count brief
touches that CHALLENGE.md says should not count. Tuning to them alone would teach a method to
flicker, so everything is also scored with short spells handed back to the team that had the ball on
both sides of them, at each length in STICKY_LEVELS. Which of these the real ground truth most
resembles is unknown, so a method should hold up across all of them rather than win on one.

Simulated Smart Tagging
-----------------------
These matches have no Smart Tagging, so the last scenario invents it from the true labels and then
makes it worse. Tags made this way are likely to be better placed than real ones, so that scenario
tests that the code path works and is an upper guess at what tags are worth. Trust the scenario
without tags more.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for folder in (os.path.join(ROOT, "evaluation"), ROOT, HERE):
    sys.path.insert(0, folder)
import possession_metrics as pm  # noqa: E402
import possession_v1  # noqa: E402
import possession_v2  # noqa: E402
from baseline import AWAY, HOME, NOBODY, VARIANTS, smooth  # noqa: E402
from possession_v1 import FPS, ball_speed, rolling_mean, tag_points  # noqa: E402

# Params
DATA_DIR = os.path.join(os.path.dirname(ROOT), "external_data", "idsse")
MATCHES = ("DFL-COM-000001_DFL-MAT-J03WMX", "DFL-COM-000001_DFL-MAT-J03WN1",
           "DFL-COM-000002_DFL-MAT-J03WOH", "DFL-COM-000002_DFL-MAT-J03WOY",
           "DFL-COM-000002_DFL-MAT-J03WPY", "DFL-COM-000002_DFL-MAT-J03WQQ",
           "DFL-COM-000002_DFL-MAT-J03WR9")
POSITIONS = "DFL_04_03_positions_raw_observed_{}.xml"
INFO = "DFL_02_01_matchinformation_{}.xml"
KMH_TO_MS = 1 / 3.6  # the IDSSE speed column is km/h, the ATD's is m/s
SEED = 2027
METHODS = (("possession_v1", possession_v1), ("possession_v2", possession_v2))
STICKY_LEVELS = (2.0, 5.0, 8.0)  # seconds: shorter spells between two of the rival's do not count

# Full-match (macroF1, spellF1) on the real ATD, from EVALUATION.md, to judge the damage against.
PUBLISHED = {"baseline_raw": (0.6532, 0.0486), "baseline_smooth25": (0.6752, 0.1996),
             "baseline_smooth50": (0.6724, 0.2433), "baseline_smooth125": (0.6356, 0.2242)}

# Damage. The first two are from DATA.md. The rest are assumptions, see the module docstring.
BALL_MISSING = 0.41
PLAYER_MISSING = 1 - 20.5 / 22
# How that 41% divides between stoppages and play, and how long the gaps last, is not published. The
# baselines' published macroF1 only makes sense if a missing ball says a lot about a dead ball, so
# these are set to bring the baselines here close to their published scores.
BALL_MISSING_DEAD = 0.8
BALL_GAP_LIVE_S = 2.0
BALL_GAP_DEAD_S = 10.0
PLAYER_GAP_S = 3.0
BALL_EDGE_BLUR_S = 3.0     # how far the hidden stretches miss the true edges of a stoppage
BALL_NOISE_M = 0.4
PLAYER_NOISE_M = 0.25
SPEED_NOISE_MS = 0.3
# A tracker's errors persist from frame to frame, so a window cannot simply average them away.
BALL_DRIFT_M = 0.5
PLAYER_DRIFT_M = 0.3
DRIFT_FRAMES = 12
# One camera on a tower sees a high ball further along the pitch than it is, because a ball in the air
# and a point on the grass behind it look the same. SLIDE_LIMIT caps that as a share of the distance.
CAMERA_BACK_M = 40.0
CAMERA_HEIGHT_M = 30.0
SLIDE_LIMIT = 0.3
WRONG_TEAM = 0.03          # chance a track picks up a player of the wrong team when it restarts
FALSE_BALL_RATE = 0.002    # chance per frame that the tracker starts following something else
FALSE_BALL_FRAMES = 10
FALSE_BALL_M = (5.0, 25.0)

# Simulated Smart Tagging. The real file has 59 set pieces and about 580 phases with a team.
SET_PIECE_RECALL = 0.5
SET_PIECE_NO_TEAM = 0.45
TAG_JITTER_S = 1.0
TAG_OPENS_S, TAG_CLOSES_S = 3.5, 1.5   # 01_smart_tagging.ipynb: opens about 3.5 s before a kickoff
PHASE_RECALL = 0.7
PHASE_DEFENDING = 0.4
PHASE_WRONG = 0.05
HOME_NAME, AWAY_NAME = "home", "away"


def attributes(line):
    """The attributes of one XML tag as a dict of strings."""
    return dict(re.findall(r'(\w+)="([^"]*)"', line))


def read_frame_sets(path):
    """Stream a positions file into (pitch, frame_sets): one dict per player or ball per half.

    Files are around 400 MB of one-frame-per-line XML, so this splits on quotes rather than building
    a document tree. Player frames are N, T, X, Y, D, S, in that order, and that is checked once per
    frame set so a change of layout fails loudly rather than reading the wrong column.
    """
    pitch, frame_sets, current = None, [], None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current["is_ball"]:
                    found = attributes(line)
                    current["rows"].append((int(found["N"]), float(found["X"]), float(found["Y"]),
                                            float(found["Z"]), int(found["BallPossession"]),
                                            int(found["BallStatus"])))
                else:
                    parts = line.split('"')
                    if not current["rows"] and parts[4] + parts[10] != " X= S=":
                        raise ValueError("{}: unexpected frame layout: {}".format(path, line[:90]))
                    current["rows"].append((int(parts[1]), float(parts[5]), float(parts[7]),
                                            float(parts[11])))
            elif line.startswith("<FrameSet "):
                found = attributes(line)
                current = {"section": found["GameSection"], "team": found["TeamId"],
                           "person": found["PersonId"], "is_ball": found["TeamId"] == "BALL",
                           "rows": []}
                frame_sets.append(current)
            elif line.startswith("<PitchSize "):
                found = attributes(line)
                pitch = (float(found["X"]), float(found["Y"]))
            elif line.startswith("</FrameSet"):
                current["rows"] = np.array(current["rows"], dtype=np.float64)
    return pitch, frame_sets


def build_match(stem, data_dir):
    """The match as arrays over one timeline, with a single ungraded frame between the halves."""
    path = os.path.join(data_dir, POSITIONS.format(stem))
    if not os.path.isfile(path):
        raise IOError("not found: {}\nSee the module docstring for where to get it.".format(path))
    with open(os.path.join(data_dir, INFO.format(stem)), encoding="utf-8") as handle:
        general = attributes(re.search(r"<General [^>]*>", handle.read()).group())
    home_team, guest_team = general["HomeTeamId"], general["GuestTeamId"]

    # Some files also track the match officials, under a team of their own. The ATD does not.
    pitch, frame_sets = read_frame_sets(path)
    frame_sets = [fs for fs in frame_sets if fs["is_ball"] or fs["team"] in (home_team, guest_team)]
    balls = {fs["section"]: fs["rows"] for fs in frame_sets if fs["is_ball"]}
    sections = sorted(balls, key=lambda section: balls[section][0, 0])  # play order

    # Where each half starts on the shared timeline, and how far its frame numbers are shifted.
    shift, blocks, n_frames = {}, [], 0
    for number, section in enumerate(sections, start=1):
        first_n, last_n = int(balls[section][0, 0]), int(balls[section][-1, 0])
        shift[section] = n_frames - first_n
        blocks.append((number, n_frames, n_frames + last_n - first_n))
        n_frames += last_n - first_n + 2

    people = sorted({(fs["team"], fs["person"]) for fs in frame_sets if not fs["is_ball"]})
    column = {person: index for index, person in enumerate(people)}
    ball = np.full((n_frames, 2), np.nan)
    ball_z = np.full(n_frames, np.nan)
    truth = np.full(n_frames, pm.X, dtype=np.int8)
    xy = np.full((n_frames, len(people), 2), np.nan)
    speed = np.full((n_frames, len(people)), np.nan)

    # Files count from the centre of the pitch, the ATD from a corner.
    corner = np.array(pitch) / 2.0
    for fs in frame_sets:
        rows = fs["rows"]
        if not len(rows):
            continue
        at = rows[:, 0].astype(np.int64) + shift[fs["section"]]
        if fs["is_ball"]:
            ball[at] = rows[:, 1:3] + corner
            ball_z[at] = rows[:, 3]
            truth[at] = np.where(rows[:, 5] == 0, pm.D, np.where(rows[:, 4] == 1, pm.H, pm.A))
        else:
            col = column[(fs["team"], fs["person"])]
            xy[at, col] = rows[:, 1:3] + corner
            speed[at, col] = rows[:, 3] * KMH_TO_MS

    team = np.array([HOME if owner == home_team else AWAY for owner, _ in people], dtype=np.int8)
    return {"name": stem[-6:], "pitch": pitch, "blocks": blocks, "ball": ball, "ball_z": ball_z,
            "truth": truth, "xy": xy, "speed": speed, "team_of": np.tile(team, (n_frames, 1))}


def load_match(stem, data_dir):
    """build_match with a cache in data_dir/cache, because parsing takes most of a minute."""
    cache = os.path.join(data_dir, "cache", stem + ".npz")
    if os.path.isfile(cache):
        with np.load(cache) as saved:
            match = {key: saved[key] for key in saved.files}
        match["name"] = str(match["name"])
        match["pitch"] = tuple(match["pitch"].tolist())
        match["blocks"] = [tuple(row) for row in match["blocks"].tolist()]
        return match

    match = build_match(stem, data_dir)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **match)
    return match


def hide_in_bursts(hazard, mean_length, rng):
    """Boolean over frames, True inside a gap. Both arguments are per frame: the chance of a gap
    starting there, and the mean length of a gap that does."""
    hidden = np.zeros(len(hazard), dtype=bool)
    draws = rng.rand(len(hazard))
    frame = 0
    while frame < len(hazard):
        if draws[frame] < hazard[frame]:
            length = 1 + int(rng.exponential(mean_length[frame]))
            hidden[frame:frame + length] = True
            frame += length
        frame += 1
    return hidden


def rate_for(fraction, mean_length):
    """Gap-start chance per frame that leaves `fraction` of frames inside gaps of that length."""
    return fraction / (1.0 - fraction) / mean_length


def slow_noise(size, spread, rng):
    """Random error of this spread that changes slowly, the way a tracker's errors do."""
    raw = rng.normal(0.0, spread * math.sqrt(2 * DRIFT_FRAMES + 1), size)
    return rolling_mean(raw, DRIFT_FRAMES)


def move_dead_ball(ball, truth, rng):
    """A copy of ball that moves during stoppages, because these files freeze it on those frames.

    It waits where it went dead, is carried to where play restarts with some wander along the way,
    and is then placed there. Each stoppage takes its own timing.
    """
    out = ball.copy()
    for start, stop, code in pm.segments(truth, pm.valid_runs(truth)):
        if code != pm.D:
            continue
        begin = ball[start]
        end = ball[stop] if stop < len(ball) and truth[stop] != pm.X else ball[stop - 1]
        where = np.linspace(0.0, 1.0, stop - start)
        leaves, arrives = rng.uniform(0.1, 0.5), rng.uniform(0.6, 0.95)
        share = np.clip((where - leaves) / (arrives - leaves), 0.0, 1.0)
        share = share * share * (3.0 - 2.0 * share)  # gather pace, then slow down
        wander = np.sin(np.pi * share)[:, np.newaxis] * rng.normal(0.0, 3.0, 2)
        out[start:stop] = begin + (end - begin) * share[:, np.newaxis] + wander
    return out


def blur_edges(mask, spread, rng):
    """A copy of mask with the start and end of each run moved at random by about `spread` frames."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        first = max(0, start + int(rng.normal(0.0, spread)))
        last = min(len(mask), stop + int(rng.normal(0.0, spread)))
        out[first:last] = True
    return out


def damage(match, seed):
    """A copy of the match that looks more like the ATD. Same seed, same damage."""
    rng = np.random.RandomState(seed)
    n_frames, n_people = match["speed"].shape
    ball = move_dead_ball(match["ball"], match["truth"], rng)

    # In play it is harder to see in a crowd, in the air and at speed, which DATA.md says is where
    # it goes missing. The split keeps the overall figure at BALL_MISSING whatever the dead-ball
    # share.
    graded = match["truth"] != pm.X
    dead = blur_edges(match["truth"] == pm.D, BALL_EDGE_BLUR_S * FPS, rng) & graded
    live = graded & ~dead
    share_dead = dead.sum() / float(dead.sum() + live.sum())
    missing_live = max(0.02, (BALL_MISSING - share_dead * BALL_MISSING_DEAD) / (1.0 - share_dead))

    gaps = match["xy"] - ball[:, np.newaxis, :]
    crowd = np.sum(np.hypot(gaps[..., 0], gaps[..., 1]) < 3.0, axis=1)
    difficulty = (1.0 + 0.5 * crowd + 2.0 * (match["ball_z"] > 2.0)
                  + 1.0 * (ball_speed(ball, 2) > 15.0))
    difficulty = difficulty / difficulty[live].mean()
    # Drawn separately and masked, so a long gap in a stoppage cannot run on into play and push the
    # overall figure past BALL_MISSING.
    gap_dead, gap_live = BALL_GAP_DEAD_S * FPS, BALL_GAP_LIVE_S * FPS
    hidden_dead = hide_in_bursts(np.full(n_frames, rate_for(BALL_MISSING_DEAD, gap_dead)),
                                 np.full(n_frames, gap_dead), rng)
    hidden_live = hide_in_bursts(rate_for(missing_live, gap_live) * difficulty,
                                 np.full(n_frames, gap_live), rng)
    ball_hidden = (dead & hidden_dead) | (live & hidden_live)

    camera = np.array([match["pitch"][0] / 2.0, -CAMERA_BACK_M])
    height = np.nan_to_num(match["ball_z"])
    slide = np.minimum(height / np.maximum(CAMERA_HEIGHT_M - height, 1.0), SLIDE_LIMIT)
    ball += (ball - camera) * slide[:, np.newaxis]
    ball += rng.normal(0.0, BALL_NOISE_M, ball.shape)
    ball += np.column_stack((slow_noise(n_frames, BALL_DRIFT_M, rng),
                             slow_noise(n_frames, BALL_DRIFT_M, rng)))
    false_ball = hide_in_bursts(np.full(n_frames, FALSE_BALL_RATE),
                                np.full(n_frames, float(FALSE_BALL_FRAMES)), rng)
    burst = np.cumsum(np.diff(np.concatenate(([0], false_ball.astype(np.int8)))) == 1)
    angle = rng.uniform(0.0, 2.0 * np.pi, burst[-1] + 1)
    reach = rng.uniform(FALSE_BALL_M[0], FALSE_BALL_M[1], burst[-1] + 1)
    ball[false_ball] += np.column_stack((reach * np.cos(angle), reach * np.sin(angle)))[
        burst[false_ball]]
    ball[ball_hidden] = np.nan

    xy = match["xy"] + rng.normal(0.0, PLAYER_NOISE_M, match["xy"].shape)
    speed = np.maximum(0.0, match["speed"] + rng.normal(0.0, SPEED_NOISE_MS, match["speed"].shape))
    team_of = match["team_of"].copy()
    gap_frames = PLAYER_GAP_S * FPS
    for col in range(n_people):
        hidden = hide_in_bursts(np.full(n_frames, rate_for(PLAYER_MISSING, gap_frames)),
                                np.full(n_frames, gap_frames), rng)
        xy[:, col, 0] += slow_noise(n_frames, PLAYER_DRIFT_M, rng)
        xy[:, col, 1] += slow_noise(n_frames, PLAYER_DRIFT_M, rng)
        xy[hidden, col] = np.nan
        speed[hidden, col] = np.nan
        restarts = np.cumsum(np.diff(np.concatenate(([0], hidden.astype(np.int8)))) == -1)
        wrong = rng.rand(restarts[-1] + 1) < WRONG_TEAM
        team_of[:, col] = np.where(wrong[restarts], 1 - team_of[:, col], team_of[:, col])

    return dict(match, ball=ball, xy=xy, speed=speed, team_of=team_of)


def signals(match):
    """(nearest, nearest_speed, crowd_speed), in the form possession_v1.read_tracking returns."""
    ball, xy, speed, team_of = match["ball"], match["xy"], match["speed"], match["team_of"]
    gaps = xy - ball[:, np.newaxis, :]
    distance = np.hypot(gaps[..., 0], gaps[..., 1])
    rows = np.arange(len(ball))
    nearest = np.full((len(ball), 2), np.nan)
    nearest_speed = np.full((len(ball), 2), np.nan)
    for side in (HOME, AWAY):
        mine = np.where((team_of == side) & ~np.isnan(distance), distance, np.inf)
        closest = mine.argmin(axis=1)
        nearest[:, side] = mine[rows, closest]
        nearest_speed[:, side] = speed[rows, closest]
    nearest[np.isnan(ball[:, 0])] = np.nan
    nearest_speed[np.isnan(ball[:, 0])] = np.nan

    seen = ~np.isnan(speed)
    counts = seen.sum(axis=1)
    crowd_speed = np.full(len(ball), np.nan)
    has_data = counts > 0
    crowd_speed[has_data] = np.where(seen, speed, 0.0).sum(axis=1)[has_data] / counts[has_data]
    return nearest, nearest_speed, crowd_speed


def fake_smart_tagging(truth, blocks, rng):
    """Events shaped like load_smart_tagging.load, invented from the truth, then made worse."""
    jitter = int(TAG_JITTER_S * FPS)
    names = {pm.H: HOME_NAME, pm.A: AWAY_NAME}
    events = []
    for _, first, last in blocks:
        parts = pm.segments(truth, [(first, last + 1)])
        for index, (start, stop, code) in enumerate(parts):
            if code == pm.D:
                if index + 1 == len(parts) or rng.rand() > SET_PIECE_RECALL:
                    continue
                team = names[parts[index + 1][2]]
                if rng.rand() < SET_PIECE_NO_TEAM:
                    team = "N/A"
                shift = rng.randint(-jitter, jitter + 1)
                events.append({"code": "SET PIECES", "tags": {"Team": team},
                               "start_frame": stop - int(TAG_OPENS_S * FPS) + shift,
                               "end_frame": stop + int(TAG_CLOSES_S * FPS) + shift})
            elif stop - start >= 3 * FPS and rng.rand() < PHASE_RECALL:
                side = code if rng.rand() > PHASE_WRONG else 1 - code
                label = "PROGRESSION"
                if rng.rand() < PHASE_DEFENDING:
                    side, label = 1 - side, "DEFENDING IN MIDDLE THIRD"
                events.append({"code": label, "tags": {"Team": names[side]},
                               "start_frame": start + rng.randint(0, 2 * FPS),
                               "end_frame": stop - rng.randint(0, 2 * FPS)})
    return events


def absorb_touches(truth, shortest_s):
    """Truth with spells shorter than shortest_s between two spells of the rival handed back."""
    runs = pm.valid_runs(truth)
    out = truth.copy()
    while True:
        parts = pm.segments(out, runs)
        changed = False
        for index in range(1, len(parts) - 1):
            start, stop, code = parts[index]
            before, after = parts[index - 1][2], parts[index + 1][2]
            if (code != pm.D and stop - start < shortest_s * FPS
                    and before == after and before not in (pm.D, code)):
                out[start:stop] = before
                changed = True
        if not changed:
            return out


def run_method(method, match, tags=None):
    """A possession script on one match: the same frame_points and best_labels its real run uses."""
    nearest, nearest_speed, crowd_speed = signals(match)
    codes = np.full(len(match["ball"]), pm.X, dtype=np.int8)
    for _, first, last in match["blocks"]:
        stop = last + 1
        points = method.frame_points(match["ball"][first:stop], nearest[first:stop],
                                     nearest_speed[first:stop], crowd_speed[first:stop],
                                     match["pitch"])
        if tags is not None:
            points = points + tags[first:stop]
        codes[first:stop] = method.best_labels(points)
    return codes


def run_baseline(match, window):
    """baseline.py on one match: nearest player to the ball, then a majority vote."""
    nearest, _, _ = signals(match)
    team = np.where(nearest[:, HOME] <= nearest[:, AWAY], HOME, AWAY).astype(np.int8)
    team[np.isnan(match["ball"][:, 0])] = NOBODY
    codes = np.full(len(team), pm.X, dtype=np.int8)
    for _, first, last in match["blocks"]:
        codes[first:last + 1] = pm.label_codes(smooth(team[first:last + 1], window))
    return codes


def label_check(match):
    """Share of close-control frames where the nearest team is the labelled one. Should be high.

    BallPossession 1 is taken to be the home side. If that were backwards this would be well under
    one half and every score below would be meaningless, so it is printed for every match.
    """
    nearest, _, _ = signals(match)
    alive = (match["truth"] == pm.H) | (match["truth"] == pm.A)
    close = alive & (np.fmin(nearest[:, HOME], nearest[:, AWAY]) < 1.0)
    side = np.where(nearest[:, HOME] <= nearest[:, AWAY], pm.H, pm.A)
    return float(np.mean(side[close] == match["truth"][close]))


def print_scenario(title, names, rows):
    """One table: spellF1 per match, then the means across matches."""
    print("\n{}".format(title))
    print("  {:<20}  {}  {:>7}  {:>7}  {:>6}  {:>6}".format(
        "", "  ".join("{:>6}".format(name) for name in names), "spellF1", "macroF1", "poss",
        "truth"))
    for label, results in rows:
        print("  {:<20}  {}  {:>7.4f}  {:>7.4f}  {:>6.0f}  {:>6.0f}".format(
            label, "  ".join("{:>6.3f}".format(r["spell_f1_mean"]) for r in results),
            np.mean([r["spell_f1_mean"] for r in results]),
            np.mean([r["macro_f1"] for r in results]),
            np.mean([r["n_pred_spells"] for r in results]),
            np.mean([r["n_gt_spells"] for r in results])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR, help="where the IDSSE files are")
    parser.add_argument("--matches", nargs="+", help="last six characters of the match ids to use")
    args = parser.parse_args()

    stems = [stem for stem in MATCHES if not args.matches or stem[-6:] in args.matches]
    names = []
    labels = [name for name, _ in METHODS] + [name for name, _ in VARIANTS]
    clean, damaged = ({label: [] for label in labels} for _ in range(2))
    sticky = {level: {label: [] for label in labels} for level in STICKY_LEVELS}
    tagged = {name: [] for name, _ in METHODS}

    for stem in stems:
        match = load_match(stem, args.data_dir)
        truth = match["truth"]
        stickier = {level: absorb_touches(truth, level) for level in STICKY_LEVELS}
        names.append(match["name"])
        graded = truth != pm.X
        print("{}: {:.0f} graded minutes, {:.1f}% dead ball, label check {:.3f}".format(
            match["name"], graded.sum() / FPS / 60.0, 100.0 * np.mean(truth[graded] == pm.D),
            label_check(match)))

        # The seed follows the match, not its place in the list, so a subset sees the same damage.
        index = MATCHES.index(stem)
        broken = damage(match, SEED + index)
        print("  after damage: ball seen on {:.1f}% of frames, {:.1f} players per frame".format(
            100.0 * np.mean(~np.isnan(broken["ball"][graded, 0])),
            float(np.mean(np.sum(~np.isnan(broken["speed"][graded]), axis=1)))))

        rng = np.random.RandomState(SEED + 1000 + index)
        events = fake_smart_tagging(truth, match["blocks"], rng)
        tags = tag_points(events, (HOME_NAME, AWAY_NAME), len(truth), match["blocks"])

        for name, method in METHODS:
            clean[name].append(pm.evaluate(truth, run_method(method, match)))
            codes = run_method(method, broken)
            damaged[name].append(pm.evaluate(truth, codes))
            for level in STICKY_LEVELS:
                sticky[level][name].append(pm.evaluate(stickier[level], codes))
            tagged[name].append(pm.evaluate(truth, run_method(method, broken, tags)))
        for name, window in VARIANTS:
            clean[name].append(pm.evaluate(truth, run_baseline(match, window)))
            codes = run_baseline(broken, window)
            damaged[name].append(pm.evaluate(truth, codes))
            for level in STICKY_LEVELS:
                sticky[level][name].append(pm.evaluate(stickier[level], codes))

    print_scenario("CLEAN tracking, no tags", names, [(n, clean[n]) for n in labels])
    print_scenario("DAMAGED tracking, no tags, labels as given", names,
                   [(n, damaged[n]) for n in labels])
    for level in STICKY_LEVELS:
        print_scenario("DAMAGED tracking, no tags, spells under {:.0f} s not counted".format(level),
                       names, [(n, sticky[level][n]) for n in labels])
    print_scenario("DAMAGED tracking, simulated Smart Tagging, labels as given", names,
                   [(n, tagged[n]) for n, _ in METHODS])

    print("\nIs the damage realistic? Baselines here against their published full-match scores:")
    print("  {:<20}  {:>10}  {:>9}  {:>10}  {:>9}".format(
        "", "spell here", "published", "macro here", "published"))
    for name, _ in VARIANTS:
        print("  {:<20}  {:>10.4f}  {:>9.4f}  {:>10.4f}  {:>9.4f}".format(
            name, np.mean([r["spell_f1_mean"] for r in damaged[name]]), PUBLISHED[name][1],
            np.mean([r["macro_f1"] for r in damaged[name]]), PUBLISHED[name][0]))
