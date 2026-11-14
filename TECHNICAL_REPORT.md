# Ball possession from broken tracking: a LightGBM model, a second look, and Viterbi

*More than a Hack 2027, Phase 1 - technical document*

**Result on the public 5 minutes:** spellF1 **0.676**, macroF1 **0.871** (baseline 0.424 / 0.581).
**Result on three unseen practice matches** (never used to train or choose anything): spellF1
**0.611**, against **0.287** for our best rule-based method.

---

## 1. The approach

For every frame we say H, A or D (X outside the two halves). The pipeline has three steps:

```
tracking (ATD) --> 99 features per frame --> LightGBM: P(H), P(A), P(D) per frame
               --> "second look" LightGBM: corrects the first model's typical mistakes
               --> Viterbi (+ Smart Tagging points): whole possessions, no flicker
```

1. **Features (99 per frame).** Where each team stands and how it moves (shape, speed, players
   near the ball), the ball's position and speed, how long the ball has been missing and where it
   was before and after the gap, the evidence of our rule-based method (possession_v2), and 13
   *restart clues*: a player at a touchline (throw-in), at a corner flag (corner), players inside
   the box, or both teams in their own halves (kick-off).
2. **First model.** A LightGBM classifier gives the chance of H, A and D on every frame.
3. **Second look.** A small second LightGBM reads the first model's answers up to 2 s either
   side and how long its labels have held, and fixes recurring mistakes (a one-frame flicker to
   the other team, a change of team drawn late).
4. **Viterbi.** It joins the frames into possessions: it picks the run of labels with the highest
   total log-probability, where every change of label pays a cost. One odd frame cannot flip the
   answer, and gaps with no evidence keep the current label.

**Why this, rather than something else**

- **Why not only rules?** The ball is missing on 41% of graded frames. Rules that use the ball have
  nothing to say there. A model can still read the players: team shape, who is pushing up, who is
  standing at a touchline for a throw-in.
- **Why LightGBM?** The data is a large table of numbers with many gaps (NaN when the ball or a
  player is not tracked). LightGBM handles missing values natively, trains fast on a laptop, and
  can be made fully deterministic, so the submission is reproducible to the byte.
- **Why Viterbi on top?** spellF1 scores whole possessions, not frames. A frame classifier
  flickers; Viterbi turns probabilities into clean possessions with only three costs to choose.
- **Why a second look?** The first model judges one frame at a time. The second one sees the
  pattern around the frame and learned from the first model's real mistakes on matches it had not
  seen.
- **Both teams treated alike.** Each half is rotated so the home team attacks towards larger x,
  and the model predicts twice (teams as they are, and swapped) and averages. So it learns what a
  team *does*, not which team it is.

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD) | **yes** | the only input to the features and models |
| Smart Tagging | **yes, lightly** | adds small points at the Viterbi step: D before a set piece, then the tagged team; a little weight to the team a phase implies; D before each kick-off |
| Homography | no | the tracking is already in pitch coordinates |
| Broadcast tracking | no | the tactical tracking covers the whole pitch |
| Videos | no | |
| Public 5-minute ground truth | **only as a final check** | never used to train, tune or choose between versions |
| **External data: IDSSE** | **yes** | 7 public professional matches with official labels: our only training data (section 3) |
| Hand labels | **none** | no frame of any game was labelled or corrected by hand |
| LLMs | **as a coding assistant only** | we used Claude (Anthropic) to help write, review and comment the code and this document. No LLM is part of the pipeline, and no match data (video, tracking, tags, ground truth) was ever given to one, as the terms require |

**Why Smart Tagging stays out of the models:** the training matches have no Smart Tagging, and
tags invented for them would teach the model to trust tags too much. So the tags are added only
at the Viterbi step, scaled to the same weight they have in our rule-based method.

## 3. Training data: other matches, damaged to look like ours

The provided game has no labels we may train on (the public 5 minutes are only a check), so the
model learns from **seven other professional matches** whose official data says who has the ball
on every frame.

| | |
|---|---|
| **Dataset** | IDSSE: 7 complete professional matches, 2022/23 Bundesliga (2) and 2. Bundesliga (5), 25 fps |
| **Source** | Bassek et al., *Scientific Data* 12, 195 (2025), doi:10.1038/s41597-025-04505-y; files from huggingface.co/datasets/pysport/idsse-data; licence CC-BY 4.0, data owner DFL |
| **Who labelled it** | the league's official data (DFL). Every tracking frame carries `BallStatus` (in play / dead) and `BallPossession` (which team). We did not label anything by hand |
| **Labels we learn** | H / A from `BallPossession`, D from `BallStatus`, read frame by frame, nothing smoothed |
| **How much** | 7 matches x 3 damaged copies, every 5th frame, both team orientations: **1.2 million rows** for the final model |
| **From the provided game?** | **No.** Nothing from the Barcelona match is used for training or tuning |

**Why we damage the training data.** The IDSSE tracking is clean: the ball is almost always
there. The competition's tracking (ATD) is not: the ball is missing on 41% of graded frames, and
players drop out. A model trained on clean data would never have learned what to do when the ball
is gone, which is exactly where this match is hardest. So before training, every IDSSE match is
broken on purpose to look like the ATD (`damage` in `starter/test_on_other_matches.py`).

**How we made it look the same.** `starter/measure_tracking.py` measures how the ATD breaks, using
only the tracking and only whole-match numbers (shares, rates, percentiles; no labels, no frame
numbers): how often the ball goes missing, for how long, where, and when it is found again. It
prints the same numbers for the damaged IDSSE matches, and the damage settings were set until the
two agreed. The result: the ball goes missing in long gaps, mostly when slow among players, and
reappears when struck; short losses are drawn in, as the ATD's tracker does; the ball position and
player speeds get a little noise; players drop out in groups; a few tracks pick up a player of the
wrong team. Two leaks are removed on purpose: the ball is never frozen during stoppages, and it does
not vanish or reappear exactly where the true labels change.

**Three different copies of each match.** Each copy gets its own random settings around the
measured values, so the model learns football and not one pattern of damage. Section 5 also tests
kinds of damage that were not in the training mix.

**Honest splits.** Frames of one match are nearly identical, so we always split by whole match.
Four matches (J03WMX, J03WOH, J03WPY, J03WQQ) were used to develop and choose; three (J03WN1,
J03WOY, J03WR9) were held back and used only to confirm. For the submission, everything is
retrained on all seven with the same recipe.

**Every setting is chosen on unseen matches.** Number of trees: early stopping with each match
left out in turn. Viterbi costs: a grid search on those left-out answers. Ties (within 0.002) go
to the setting that does best across four versions of the labels (see section 4), because we do
not know how the real ground truth treats brief touches.

## 4. Possession rules we implemented

- **Who has the ball:** the team that controls it, as learned from the official labels. A change
  of team must be supported by enough evidence to pay Viterbi's switch cost (chosen: 8, in log
  points, after the second look). A single frame never flips possession.
- **Loose balls:** while the ball is in the air, missing, or nobody clearly controls it, the
  possession stays with the last team. Viterbi only changes it once the other team clearly has it.
- **Challenges and deflections:** a contest stays with the team that had the ball. A touch that
  bounces straight back earns too little to pay for two changes (there and back), which matches
  the challenge's rule that a touch without control is not a change of possession.
- **Dead ball and restarts:** D while play is stopped (ball out, stillness while players walk,
  restart positions); D before each half's kick-off tag; after a set piece the restart goes to the
  team Smart Tagging names (throw-ins are tagged without a team, so they only argue for D).
- **X:** only outside the two halves. We never abstain inside play.
- **Brief touches:** the official training labels change team much more often than the
  challenge's definition suggests. So every choice was scored on four versions of the labels: as
  given, and with spells under 2, 5 and 8 s handed back to the rival. A setting had to hold up on
  all four.

## 5. Results

**Held-back practice matches** (the trial model, trained only on the four tuning matches; spellF1
as given / brief touches under 2 s, 5 s, 8 s handed back):

| damage to the tracking | model | + second look | possession_v2 (rules) |
|---|---|---|---|
| default (measured on the ATD) | 0.613 / 0.668 / 0.531 / 0.447 | 0.611 / 0.674 / 0.542 / 0.455 | 0.287 / 0.364 / 0.420 / 0.391 |
| ball lost twice as often | 0.596 / 0.654 / 0.523 / 0.439 | 0.598 / 0.670 / 0.540 / 0.457 | 0.262 / 0.339 / 0.386 / 0.372 |
| ball found half as fast | 0.588 / 0.647 / 0.520 / 0.436 | 0.606 / 0.670 / 0.540 / 0.455 | 0.280 / 0.355 / 0.398 / 0.372 |
| many wrong-team tracks | 0.574 / 0.620 / 0.487 / 0.404 | 0.575 / 0.627 / 0.509 / 0.419 | 0.207 / 0.259 / 0.292 / 0.271 |
| no ball at all (stress test) | 0.181 / 0.203 / 0.209 / 0.168 | 0.150 / 0.170 / 0.174 / 0.144 | 0.085 / 0.113 / 0.176 / 0.189 |

A leak check (the stoppage path started about a second off its true frame) gave 0.611 as given,
so the model does not rely on exact timing that only the true answers have.

**Submission model, all seven matches, each left out in turn:** first model 0.616 as given; with
the second look 0.625 (+0.009), better on all four versions and on all seven matches.

**Public 5 minutes of Barcelona - Athletic Club** (a check only; about +/-0.04 noise with 17
possessions):

| method | macroF1 | spellF1 |
|---|---|---|
| **possession_model (submission)** | **0.871** | **0.676** |
| possession_v1 (rules) | 0.732 | 0.457 |
| possession_v2 (tuned rules) | 0.722 | 0.427 |
| baseline_smooth125 | 0.581 | 0.424 |

**What we kept, and what we dropped.** Every idea had its pass rule written down before it was
run: first on the tuning matches (average gain of at least +0.005 and no version worse), then once
on the held-back matches.

| idea | result | kept? |
|---|---|---|
| second look | +0.007 as given, +0.008 to +0.011 at 2/5/8 s (held back) | **yes** |
| restart clues (throw-ins, corners, kick-offs) | +0.014 as given, +0.005 to +0.009 at 2/5/8 s (held back) | **yes** |
| learning labels with brief touches removed | 0.539 vs 0.594 as given | no |
| a third model that makes changes of team cheaper where likely | 0.594 vs 0.594 | no |
| averaging several random seeds | +0.002, inside the noise | no |
| self-training on the model's confident answers | -0.001 | no |
| tracking facts given to the second look | +0.004, under the bar | no |
| a third look | -0.004 | no |
| second look trained on more kinds of damage | -0.0004 | no |
| goal-kick clues / free-kick wall clues | -0.006 / -0.001 | no |

## 6. Limitations, and what we would do with more time

- **It needs the ball tracking.** With no ball at all it falls to about 0.15-0.18, and the second
  look makes that case worse.
- **Short possessions are missed.** On the public 5 minutes it finds 12 possessions where there
  are 17: very short spells are merged into the possessions around them.
- **Different labels.** The training labels come from another league, another tracking system and
  another definition of possession. The four label versions reduce the risk; they do not remove it.
- **The damage is partly assumed.** Wrong-team tracks and how a high ball looks to the camera
  cannot be measured from the ATD alone.
- **Known hard cases:** a goalkeeper holding the ball while everyone walks looks like a dead ball,
  and a ball carried to a restart looks alive until it stops.
- **With more time:** more training matches from other leagues; a sequence model (for example a
  small temporal network) instead of the second look; using the broadcast tracking when the ball is
  missing from the tactical one; set-piece detection from the video.

## 7. Reproduction

Python 3.11. All randomness is seeded and LightGBM runs in deterministic mode, so the same commands
give the same model and the same file.

```bash
pip install -r requirements-model.txt          # numpy, pandas, scipy, lightgbm, pinned

# The match data goes in ./data (or set MTAH_DATA_DIR), as in DATA.md.

# A) Rebuild the submission file from the trained models in models/ (a few minutes)
python starter/possession_model.py             # -> evaluation/results/possession_model.csv
python evaluation/validate_submission.py evaluation/results/possession_model.csv
python evaluation/run_evaluation.py --no-detail

# B) Retrain the models from scratch (about 45 minutes)
#    IDSSE files from huggingface.co/datasets/pysport/idsse-data into ../external_data/idsse
#    (or set IDSSE_DIR)
python starter/train_possession_model.py features --all
python starter/train_possession_model.py cv --all
python starter/train_possession_model.py costs --all
python starter/train_possession_model.py second --all
python starter/train_possession_model.py final --all      # writes models/

# C) The held-back test in section 5 (about 40 minutes)
python starter/train_possession_model.py features         # then cv, costs, second, final
python starter/train_possession_model.py test
```

| file | what it does |
|---|---|
| `starter/possession_model.py` | the submission: features, both models, Viterbi |
| `starter/train_possession_model.py` | training, choice of settings, held-back test |
| `starter/test_on_other_matches.py` | reads the IDSSE matches and damages their tracking |
| `starter/possession_v1.py`, `possession_v2.py` | the rule-based methods; their code is reused by the model |
| `starter/measure_tracking.py` | measured how the ATD breaks (used once to set the damage) |
| `models/` | the trained models and their settings |
