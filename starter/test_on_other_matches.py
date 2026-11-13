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
IDSSE: seven complete matches from the 2022/23 Bundesliga (two) and 2. Bundesliga (five) at 25 fps,
licensed CC-BY 4.0, data owner DFL.
Bassek et al., Scientific Data 12, 195 (2025), https://doi.org/10.1038/s41597-025-04505-y

Every frame of the ball says whether it is in play (BallStatus) and which team has it
(BallPossession), which is everything a frame,label ground truth needs. Download the files into
--data-dir, which defaults to a folder beside this repository so nothing large can be committed:

    https://huggingface.co/datasets/pysport/idsse-data

Why the tracking gets damaged first
-----------------------------------
This tracking is clean and the competition's is not. Scored as it comes, any method would look far
better than it will on the real thing, so each match is also scored after making it look like the
ATD. How the ATD breaks was measured with measure_tracking.py, which reads only the tracking and
prints whole-match numbers, and the settings under Damage are the ones that make it print the same
numbers for these matches. The ball goes missing less often than chance would have it but for much
longer, mostly when it is slow among players, and is found again when it is struck. Losses under
about four seconds the tracker draws in itself. Positions are smooth from frame to frame, and
players go missing in groups, mostly while the ball is missing too.

Two things have to go whatever else is done, because they give the answer away. These files freeze
the ball on every frame of a stoppage, so damage moves it the way a ball really moves then. And the
ball must not vanish or reappear exactly where the true labels change, so 'in a stoppage', which
makes a loss likelier, is deliberately out of line with them.

What the measurements cannot show, such as tracks that swap to the wrong team or how a high ball
looks from the camera, is an assumption. A different league, tracking system and definition of
possession mean this is evidence and never a prediction.

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
from possession_v1 import FPS, tag_points  # noqa: E402

# Params
DATA_DIR = os.environ.get("IDSSE_DIR") or os.path.join(os.path.dirname(ROOT), "external_data", "idsse")
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

# Damage, set so that measure_tracking.py prints the same numbers for these matches as for the ATD
# (see the module docstring). train_possession_model.py draws each one from a range around these.
#
# The ball. Chances per frame of the tracker losing it, and of finding it again once lost, times a
# factor for its situation: its speed over the SPEED_LAG frames before or after (under 3, 3-10, 10-20
# and over 20 m/s), the crowd round it and where it is. They are set so that gaps start and end in
# the same situations as the ATD's, as measure_tracking.py counts them.
SPEED_LAG = 4
SPEEDS = (3.0, 10.0, 20.0)
LOSS_RATE = 0.00042
LOSS_BY_SPEED = (1.9, 2.0, 0.001, 0.001)
LOSS_DEAD = 4.6            # in a stoppage
LOSS_IN_CROWD = 1.1        # with 3 or more players within 3 m
LOSS_IN_BOX = 1.6          # in a penalty box, on top of being at an end of the pitch
LOSS_ACROSS = (3.1, 1.1, 0.19)   # by thirds of the width, from y = 0
LOSS_ALONG = (0.72, 0.74, 2.8)   # by thirds of the length, from x = 0
FIND_RATE = 0.025
FIND_BY_SPEED = (0.03, 0.26, 1.1, 39.0)
FIND_DEAD = 1.5
# The ATD seldom shows the ball off the pitch, but never loses it at speed, so it is hidden there
# once it is more than OUTSIDE_M out and slower than OUTSIDE_SLOW_MS: lying, or carried to a restart.
OUTSIDE_M = 1.0
OUTSIDE_SLOW_MS = 3.0
# The ATD has no gaps of 1 to 5 seconds: the tracker draws shorter losses in itself. Here that is a
# smooth curve from where it lost the ball to where it found it again, right only where the ball
# really went that way, except on the FILL_TRUE share where the tracker gets it right.
FILL_S = 5.0
FILL_TRUE = 0.0
FILL_SPEED_CAP = 10.0      # m/s, the most it lets the drawn-in ball leave or arrive at
# Blips of under a second, mostly as the ball is struck, which the tracker leaves as gaps. The chance
# per frame is GLITCH_RATE times how sharply the ball changes velocity there.
GLITCH_RATE = 0.53
GLITCH_FRAMES = 6.9
BALL_EDGE_BLUR_S = 3.0     # how far 'in a stoppage' misses the true edges of one
ROLL_S = (0.5, 2.0)        # a ball that goes dead rolls on for this long before it stops
ROLL_MAX_MS = 25.0
# The path drawn for a stoppage starts on its true first frame, which a model could learn to spot.
# Above zero, the start moves by about this many seconds either way, from a stream of its own so that
# every other draw stays as it was. Only train_possession_model.py's 'test' sets it, as a leak check.
DEAD_START_SHIFT_S = 0.0
# The ATD's positions are smoother from frame to frame than these files', ball and players alike,
# so they are smoothed over about this many frames.
BALL_SMOOTH_FRAMES = 8.0
PLAYER_SMOOTH_FRAMES = 2.0
BALL_NOISE_M = 0.002
# A tracker's errors persist from frame to frame, so a window cannot simply average them away.
BALL_DRIFT_M = 0.11
DRIFT_FRAMES = 12
# One camera on a tower sees a high ball further along the pitch than it is, because a ball in the air
# and a point on the grass behind it look the same. SLIDE_LIMIT caps that as a share of the distance.
CAMERA_BACK_M = 40.0
CAMERA_HEIGHT_M = 30.0
SLIDE_LIMIT = 0.3
FALSE_BALL_RATE = 0.0      # chance per frame that the tracker starts following something else
FALSE_BALL_FRAMES = 10
FALSE_BALL_M = (5.0, 25.0)

# Players. Each goes missing now and then, more often with another player within CLOSE_M, and on
# some of the ball's longer gaps a group goes with it. A few are tracked twice for a moment.
PLAYER_LOSS_RATE = 0.000032
PLAYER_LOSS_CLOSE = 150.0
CLOSE_M = 1.0
PLAYER_GAP_S = 5.5
GROUP_LOSS = 0.6           # chance that a gap in the ball of FILL_S or more takes a group with it
GROUP_SIZES = (2, 20)      # players in such a group, the chance of each size falling as size^-1.5
GROUP_SIZE_POWER = 1.5
GHOST_RATE = 0.00077
GHOST_S = 2.3
GHOST_M = (0.5, 1.5)
PLAYER_NOISE_M = 0.0
PLAYER_DRIFT_M = 0.1
PLAYER_DRIFT_FRAMES = 50
SPEED_NOISE_MS = 0.16
WRONG_TEAM = 0.03          # chance a track picks up a player of the wrong team when it restarts

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


def runs(mask):
    """(starts, stops) of each run of True in a 1-D mask, stops exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def slow_noise(size, spread, frames, rng):
    """Random error of this spread that changes smoothly over about `frames`, the way a tracker's
    errors do. Smooth rather than stepped, so that it does not show as shaking from frame to frame."""
    if spread <= 0.0:
        return np.zeros(size)
    kernel = np.exp(-0.5 * (np.arange(-3 * frames, 3 * frames + 1) / float(frames)) ** 2)
    kernel /= math.sqrt(np.sum(kernel ** 2))
    return spread * np.convolve(rng.normal(0.0, 1.0, size), kernel, mode="same")


def gaussian(frames):
    """A Gaussian kernel over +/- 3 standard deviations of `frames`, summing to one."""
    kernel = np.exp(-0.5 * (np.arange(-int(3 * frames), int(3 * frames) + 1) / float(frames)) ** 2)
    return kernel / kernel.sum()


def smoothed(values, frames, blocks):
    """values (frames first, any shape after) smoothed over about `frames`, half by half, with NaN
    left out of every average and kept as NaN."""
    if frames <= 0.0:
        return values.copy()
    kernel = gaussian(frames)
    flat = values.reshape(len(values), -1)
    out = flat.copy()
    for _, first, last in blocks:
        for col in range(flat.shape[1]):
            part = flat[first:last + 1, col]
            known = ~np.isnan(part)
            if len(part) < len(kernel) or not known.any():
                continue
            weight = np.convolve(known.astype(np.float64), kernel, mode="same")
            total = np.convolve(np.where(known, part, 0.0), kernel, mode="same")
            out[first:last + 1, col] = np.where(known, total / np.maximum(weight, 1e-9), np.nan)
    return out.reshape(values.shape)


def dead_path_start(truth, start, stop, previous, shift_rng):
    """Where the drawn path of the stoppage [start, stop) begins, moved by DEAD_START_SHIFT_S.

    It never begins before SPEED_LAG frames of the play before it, nor past halfway through.
    """
    spread = DEAD_START_SHIFT_S * FPS
    first = start + int(round(np.clip(shift_rng.normal(0.0, spread), -2.0 * spread, 2.0 * spread)))
    first = min(max(first, previous + SPEED_LAG + 1), start + (stop - start) // 2)
    if first < start and np.any(truth[first - SPEED_LAG:start] == pm.X):
        return start
    return first


def move_dead_ball(ball, truth, rng, shift_rng=None):
    """A copy of ball that moves during stoppages, because these files freeze it on those frames.

    It rolls on from where it went dead and comes to rest, waits, is carried to where play restarts
    with some wander along the way, and is then placed there. Each stoppage takes its own timing.
    With shift_rng the rolling starts off the true first frame: early, over the last of the play, or
    late, with the ball carrying on as it was going until then. The restart is the real ball being
    struck, so its end stays where it is.
    """
    out = ball.copy()
    previous = 0
    for start, stop, code in pm.segments(truth, pm.valid_runs(truth)):
        if code != pm.D:
            previous = start
            continue
        first = start if shift_rng is None else dead_path_start(truth, start, stop, previous,
                                                                 shift_rng)
        anchor = min(first, start)  # the last frame the path takes the ball's motion from
        begin = ball[anchor]
        end = ball[stop] if stop < len(ball) and truth[stop] != pm.X else ball[stop - 1]
        velocity = np.zeros(2)
        if anchor >= SPEED_LAG and truth[anchor - SPEED_LAG] != pm.X:
            velocity = np.nan_to_num((begin - ball[anchor - SPEED_LAG]) * FPS / float(SPEED_LAG))
        if first > start:
            out[start:first] = begin + np.outer(np.arange(first - start) / float(FPS), velocity)
            begin = begin + velocity * (first - start) / float(FPS)
        pace = math.hypot(velocity[0], velocity[1])
        if pace > ROLL_MAX_MS:
            velocity = velocity * ROLL_MAX_MS / pace
        roll = rng.uniform(*ROLL_S)
        rolling = np.minimum(np.arange(stop - first) / float(FPS), roll)
        rolled = begin + np.outer(rolling - rolling * rolling / (2.0 * roll), velocity)
        where = np.linspace(0.0, 1.0, stop - first)
        leaves, arrives = rng.uniform(0.1, 0.5), rng.uniform(0.6, 0.95)
        share = np.clip((where - leaves) / (arrives - leaves), 0.0, 1.0)
        share = share * share * (3.0 - 2.0 * share)  # gather pace, then slow down
        wander = np.sin(np.pi * share)[:, np.newaxis] * rng.normal(0.0, 3.0, 2)
        out[first:stop] = rolled + (end - rolled[-1]) * share[:, np.newaxis] + wander
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


def ball_speeds(ball):
    """(before, after): the ball's speed in m/s over the SPEED_LAG frames up to each frame and on
    from it, as measure_tracking.py takes it. NaN where it is not known."""
    before = np.full(len(ball), np.nan)
    moved = ball[SPEED_LAG:] - ball[:-SPEED_LAG]
    before[SPEED_LAG:] = np.hypot(moved[:, 0], moved[:, 1]) * FPS / float(SPEED_LAG)
    after = np.full(len(ball), np.nan)
    after[:-SPEED_LAG] = before[SPEED_LAG:]
    return before, after


def by_speed(speed, factors):
    """factors[i] for the SPEEDS group of each speed. An unknown speed counts as the slowest."""
    return np.asarray(factors, dtype=np.float64)[np.digitize(np.nan_to_num(speed), SPEEDS)]


def lose_and_find(start, find, rng):
    """Boolean over frames, True while the ball is lost. A loss starts on a frame with chance
    start there, and ends on each frame after with chance find there."""
    starting, finding = rng.rand(len(start)).tolist(), rng.rand(len(start)).tolist()
    start, find = start.tolist(), find.tolist()
    lost, losing = [False] * len(start), False
    for frame in range(len(start)):
        losing = finding[frame] >= find[frame] if losing else starting[frame] < start[frame]
        lost[frame] = losing
    return np.array(lost, dtype=bool)


def capped(velocity):
    """velocity, slowed if need be to FILL_SPEED_CAP."""
    pace = math.hypot(velocity[0], velocity[1])
    return velocity * (FILL_SPEED_CAP / pace) if pace > FILL_SPEED_CAP else velocity


def draw_in(ball, start, stop):
    """The tracker's line over a loss from start to stop: a smooth curve from the last place it saw
    the ball to the next, leaving and arriving at the speeds the ball had there."""
    lag, last = SPEED_LAG, len(ball) - 1
    seen_last, seen_next = ball[start - 1], ball[stop]
    leaving = capped(np.nan_to_num((seen_last - ball[max(start - 1 - lag, 0)]) * FPS / float(lag)))
    arriving = capped(np.nan_to_num((ball[min(stop + lag, last)] - seen_next) * FPS / float(lag)))
    span = (stop - start + 1) / float(FPS)
    s = (np.arange(start, stop) - (start - 1))[:, np.newaxis] / float(stop - start + 1)
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * seen_last + (s ** 3 - 2 * s ** 2 + s) * span * leaving
            + (3 * s ** 2 - 2 * s ** 3) * seen_next + (s ** 3 - s ** 2) * span * arriving)


def damage_ball(match, ball, rng):
    """(ball as the tracker reports it, True where it reports none), from the true ball with its
    stoppages moved."""
    n_frames = len(ball)
    length, width = match["pitch"]
    graded = match["truth"] != pm.X
    dead = blur_edges(match["truth"] == pm.D, BALL_EDGE_BLUR_S * FPS, rng) & graded
    # What the tracker knows of the ball's motion is smoothed like the rest of what it reports.
    steady = smoothed(ball, BALL_SMOOTH_FRAMES, match["blocks"])
    before, after = ball_speeds(steady)
    gaps = match["xy"] - ball[:, np.newaxis, :]
    crowd = np.sum(np.hypot(gaps[..., 0], gaps[..., 1]) < 3.0, axis=1)
    x, y = ball[:, 0], ball[:, 1]
    in_box = ((x < 16.5) | (x > length - 16.5)) & (np.abs(y - width / 2.0) < 20.16)
    edge = OUTSIDE_M
    off_pitch = (((x < -edge) | (x > length + edge) | (y < -edge) | (y > width + edge))
                 & (np.nan_to_num(before) < OUTSIDE_SLOW_MS))
    across = np.digitize(np.nan_to_num(y, nan=width / 2.0), (width / 3.0, 2.0 * width / 3.0))
    along = np.digitize(np.nan_to_num(x, nan=length / 2.0), (length / 3.0, 2.0 * length / 3.0))
    start = (LOSS_RATE * by_speed(before, LOSS_BY_SPEED) * np.where(dead, LOSS_DEAD, 1.0)
             * np.where(crowd >= 3, LOSS_IN_CROWD, 1.0) * np.where(in_box, LOSS_IN_BOX, 1.0)
             * np.asarray(LOSS_ACROSS)[across] * np.asarray(LOSS_ALONG)[along])
    find = FIND_RATE * by_speed(after, FIND_BY_SPEED) * np.where(dead, FIND_DEAD, 1.0)
    lost = lose_and_find(np.clip(start, 0.0, 1.0) * graded, np.clip(find, 0.0, 1.0), rng) & graded

    # Smoothed after the camera's view of a high ball is added, as the tracker smooths what it sees.
    camera = np.array([length / 2.0, -CAMERA_BACK_M])
    height = np.nan_to_num(match["ball_z"])
    slide = np.minimum(height / np.maximum(CAMERA_HEIGHT_M - height, 1.0), SLIDE_LIMIT)
    seen = smoothed(ball + (ball - camera) * slide[:, np.newaxis], BALL_SMOOTH_FRAMES,
                    match["blocks"])

    # Losses short enough are drawn in, and stay gaps only where they reach the end of a half.
    out, hidden = seen.copy(), lost.copy()
    for _, first, last in match["blocks"]:
        starts, stops = runs(lost[first:last + 1])
        for begin, end in zip(starts + first, stops + first):
            if end - begin >= FILL_S * FPS or begin == first or end > last:
                continue
            hidden[begin:end] = False
            if rng.rand() >= FILL_TRUE:
                out[begin:end] = draw_in(seen, begin, end)

    live = graded & ~dead
    hidden |= hide_in_bursts(GLITCH_RATE * possession_v2.touch_strength(steady) * live,
                             np.full(n_frames, GLITCH_FRAMES), rng) & graded
    hidden |= off_pitch & graded

    out += rng.normal(0.0, BALL_NOISE_M, out.shape)
    out += np.column_stack((slow_noise(n_frames, BALL_DRIFT_M, DRIFT_FRAMES, rng),
                            slow_noise(n_frames, BALL_DRIFT_M, DRIFT_FRAMES, rng)))
    false_ball = hide_in_bursts(np.full(n_frames, FALSE_BALL_RATE),
                                np.full(n_frames, float(FALSE_BALL_FRAMES)), rng)
    burst = np.cumsum(np.diff(np.concatenate(([0], false_ball.astype(np.int8)))) == 1)
    angle = rng.uniform(0.0, 2.0 * np.pi, burst[-1] + 1)
    reach = rng.uniform(FALSE_BALL_M[0], FALSE_BALL_M[1], burst[-1] + 1)
    out[false_ball] += np.column_stack((reach * np.cos(angle), reach * np.sin(angle)))[
        burst[false_ball]]
    out[hidden] = np.nan
    return out, hidden


def nearest_other(xy, chunk=4000):
    """(n, p) distance from each player to the nearest other one. inf where it is not known."""
    n_frames, n_people, _ = xy.shape
    others = ~np.eye(n_people, dtype=bool)
    out = np.full((n_frames, n_people), np.inf)
    for first in range(0, n_frames, chunk):
        part = xy[first:first + chunk]
        gaps = part[:, :, np.newaxis, :] - part[:, np.newaxis, :, :]
        distance = np.hypot(gaps[..., 0], gaps[..., 1])
        out[first:first + chunk] = np.where(others & ~np.isnan(distance), distance,
                                            np.inf).min(axis=2)
    return out


def damage_players(match, ball_hidden, rng):
    """(xy, speed, team_of) as the tracker reports them, with one more slot for a player it tracks
    twice for a moment."""
    xy = smoothed(match["xy"], PLAYER_SMOOTH_FRAMES, match["blocks"])
    speed, team_of = match["speed"].copy(), match["team_of"].copy()
    n_frames, n_people = speed.shape
    close = nearest_other(match["xy"]) < CLOSE_M
    gap_frames = np.full(n_frames, PLAYER_GAP_S * FPS)
    hidden = np.zeros((n_frames, n_people), dtype=bool)
    for col in range(n_people):
        hidden[:, col] = hide_in_bursts(
            PLAYER_LOSS_RATE * np.where(close[:, col], PLAYER_LOSS_CLOSE, 1.0), gap_frames, rng)
    sizes = np.arange(GROUP_SIZES[0], GROUP_SIZES[1] + 1)
    weights = sizes ** -GROUP_SIZE_POWER / np.sum(sizes ** -GROUP_SIZE_POWER)
    for begin, end in zip(*runs(ball_hidden)):
        if end - begin < FILL_S * FPS or rng.rand() >= GROUP_LOSS:
            continue
        keep = rng.uniform(0.3, 1.0)
        first = begin + int(rng.uniform(0.0, 1.0 - keep) * (end - begin))
        last = first + max(1, int(keep * (end - begin)))
        present = np.flatnonzero(~np.isnan(xy[first, :, 0]))
        size = min(len(present), int(rng.choice(sizes, p=weights)))
        hidden[first:last, rng.choice(present, size, replace=False)] = True

    for col in range(n_people):
        xy[:, col, 0] += slow_noise(n_frames, PLAYER_DRIFT_M, PLAYER_DRIFT_FRAMES, rng)
        xy[:, col, 1] += slow_noise(n_frames, PLAYER_DRIFT_M, PLAYER_DRIFT_FRAMES, rng)
        restarts = np.cumsum(np.diff(np.concatenate(([0], hidden[:, col].astype(np.int8)))) == -1)
        wrong = rng.rand(restarts[-1] + 1) < WRONG_TEAM
        team_of[:, col] = np.where(wrong[restarts], 1 - team_of[:, col], team_of[:, col])
    xy += rng.normal(0.0, PLAYER_NOISE_M, xy.shape)
    speed = np.maximum(0.0, speed + rng.normal(0.0, SPEED_NOISE_MS, speed.shape))
    xy[hidden] = np.nan
    speed[hidden] = np.nan

    ghost_xy = np.full((n_frames, 1, 2), np.nan)
    ghost_speed = np.full((n_frames, 1), np.nan)
    ghost_team = np.zeros((n_frames, 1), dtype=team_of.dtype)
    spells = hide_in_bursts(np.full(n_frames, GHOST_RATE), np.full(n_frames, GHOST_S * FPS), rng)
    for begin, end in zip(*runs(spells)):
        present = np.flatnonzero(~np.isnan(xy[begin, :, 0]))
        if not len(present):
            continue
        source = present[rng.randint(len(present))]
        angle, reach = rng.uniform(0.0, 2.0 * np.pi), rng.uniform(*GHOST_M)
        ghost_xy[begin:end, 0] = xy[begin:end, source] + reach * np.array([np.cos(angle),
                                                                            np.sin(angle)])
        ghost_speed[begin:end, 0] = speed[begin:end, source]
        ghost_team[begin:end, 0] = team_of[begin:end, source]
    return (np.concatenate((xy, ghost_xy), axis=1), np.concatenate((speed, ghost_speed), axis=1),
            np.concatenate((team_of, ghost_team), axis=1))


def damage(match, seed):
    """A copy of the match that looks like the ATD. Same seed, same damage."""
    rng = np.random.RandomState(seed)
    shift_rng = np.random.RandomState([seed, 1]) if DEAD_START_SHIFT_S else None
    ball = move_dead_ball(match["ball"], match["truth"], rng, shift_rng)
    ball, hidden = damage_ball(match, ball, rng)
    xy, speed, team_of = damage_players(match, hidden, rng)
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
