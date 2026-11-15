# HalaNJU | More than a Hack 2027

Frame-level ball possession for the Barcelona vs Athletic Club tactical recording. The program writes home possession (H), away possession (A), stoppage (D) or ungraded (X) for each of 199,319 video frames. The classifiers in `possession/idsse_models.joblib` were fitted on seven external matches. Inference reads the tactical tracking and the automatic Smart Tagging file of the competition match. The match data is gated and is absent from this repository.

## Reproduce the submission

Use CPython 3.13.11. Place `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_smart_tagging.xml` in `data/tactical/`, or pass an existing data bundle with `--data-dir /path/to/data`.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r possession/requirements.txt
export LOKY_MAX_CPU_COUNT=1
.venv/bin/python -m possession.predict --output HalaNJU.csv
.venv/bin/python evaluation/validate_submission.py HalaNJU.csv
shasum -a 256 HalaNJU.csv
```

The expected SHA-256 is `58bf020b0b7b0db03441943d4ba83f6dd288af17f950b738e8f5080155cdaa3b`. The file is UTF-8 with LF line endings and contains 316 predicted possessions.

## Rebuild the models and the validation results

The download needs `curl` and about 2.63 GB of storage. The script reads Figshare version 1 of the dataset and verifies the published byte count and MD5 of each XML file.

```sh
W=work
.venv/bin/python -m possession.fetch_idsse $W/idsse
.venv/bin/python -m possession.prepare --idsse $W/idsse --cache $W/features --interactions
.venv/bin/python -m possession.interaction_train --cache $W/features --folds $W/folds --workers 9
.venv/bin/python -m possession.interaction_validation --cache $W/features --folds $W/folds --work $W/validation --report $W/experiment.json --workers 9
.venv/bin/python -m possession.control_scores --cache $W/features --work $W/validation --output $W/control.json
.venv/bin/python -m possession.interaction_train --cache $W/features --model $W/retrained.joblib --calibration $W/experiment.json
.venv/bin/python -m possession.predict --model $W/retrained.joblib --output $W/retrained.csv
cmp $W/retrained.csv HalaNJU.csv
```

`validation/experiment.json` and `validation/control.json` hold the outputs of the validation commands for the bundled model. The second training command takes about 25 minutes with nine processes.

## Method

**Training conditions.** The target ball track is smoothed, and a ball that stays still for long is usually absent from it. The external ball tracks are sharp and freeze in place during stoppages. `possession/prepare.py` therefore smooths every external ball track with a 7-frame Gaussian and writes four conditions per match: the target's visibility sequence replayed on the external half, full visibility, both of these again with the ball hidden wherever it stays still for 5 s or longer. All four conditions train the classifiers.

**Classifiers.** One gradient-boosted classifier estimates which team has the ball and another estimates stoppages. Inputs are distances between players and ball, counts of players within 2, 5 and 10 m, closing speeds, team shape and pace. When the ball is hidden, a carrier-path model scores player trajectories. The other half of the same match supplies automatic examples of clear touches. [PathCRF, inspiration for carrier paths](https://arxiv.org/html/2602.12080v1)

**Decoder.** A 5 Hz decoder tracks elapsed possession time on each candidate path. Continuation, turnover and stoppage rates by possession age come from the training matches. A direct change of team costs 1.5 log-score units with a visible ball and 3.5 in occlusion. Stoppages last at least 1 s.

**Controlled possession.** The external labels change team at any opposing touch. The competition counts a change of team only for controlled possession. A possession that begins with a direct change of team and lasts under 1.5 s is merged into the preceding team's possession (`CONTROL_SECONDS` in `possession/predict.py`).

## Validation

Nested leave-one-match-out validation on the seven external matches. Six matches train the classifiers for each test match, and an inner leave-one-out over those six selects the switch costs. Mean spellF1 over the seven test matches:

| Test condition | External labels | Reference merged below 1.5 s |
|---|---:|---:|
| Replayed occlusions, decoder output | 0.5744 | 0.6404 |
| Replayed occlusions, merged below 1.5 s | 0.5699 | 0.6690 |
| Full visibility, decoder output | 0.6636 | 0.7069 |
| Full visibility, merged below 1.5 s | 0.6529 | 0.7640 |

Smooth50 scores 0.1986 with replayed occlusions. The public five-minute excerpt of the target match was used to check frame alignment and file format. It was excluded from training, tuning and model selection.

## Data and code credits

Bassek, Rein, Weber and Memmert published the seven matches under CC BY 4.0. DFL and Sportec supplied the tracking and the per-frame reference labels. The code maps `BallStatus=0` to D and active-ball `BallPossession=1/2` to H/A. [Versioned data release](https://doi.org/10.6084/m9.figshare.28196177.v1) · [Dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)

Metrica Sports and the competition organisers supply `games.py`, the two `starter` helpers and the scoring and validation helpers in `evaluation`. Their MIT notice accompanies the code. The competition match assets remain subject to the terms of enrolment. [Official competition repository](https://github.com/metrica-sports/more-than-a-hack-2027)
