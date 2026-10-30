# ASL Alphabet Detection (laptop prototype)

Real-time American Sign Language **fingerspelling** recognition from a webcam,
built as the laptop-side prototype for an AlphaMini robot ASL reader.

## How it works

```
  Webcam  ->  MediaPipe Hands  ->  63 normalized numbers  ->  classifier  ->  letter
            (finds 21 points        (position/size              (Random
             on your hand)           independent)                Forest)
```

Instead of classifying raw images of the hand, we let MediaPipe find 21 landmark
points (knuckles, fingertips, wrist) and classify the **shape** those points
make. This is lightweight, runs in real time on a CPU, learns *your* hands in
seconds, and ports cleanly to the robot later (the AlphaMini's camera frames go
through the exact same pipeline).

## Setup (one time)

First install **Python 3.10, 3.11, or 3.12** from
[python.org](https://www.python.org/downloads/) — during install, tick
*"Add python.exe to PATH"*. (MediaPipe does not support Python 3.13 yet.)

**Easiest (Windows):** double-click **`setup.bat`**. It finds your Python,
creates the `.venv` environment, and installs everything. Then jump to *Run it*.

**Or set it up manually:**

```powershell
# from the asl-detection folder
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Note: the `.venv` folder is **not** included in the shared zip because it is
> tied to the machine that created it — that's why you run `setup.bat` once on a
> new PC. A trained model (`model/asl_model.joblib`) *is* included, so detection
> works right away; retrain with `run_1_collect.bat` + `run_2_train.bat` to tune
> it to your own hands.

## Run it (3 steps)

**Easiest:** double-click the launchers in order — `run_1_collect.bat`,
`run_2_train.bat`, `run_3_detect.bat`. (Or run the commands below in an
activated venv.)

```powershell
python collect_data.py   # 1. record yourself signing each letter
python train_model.py    # 2. train the model (a few seconds)
python detect_asl.py     # 3. live detection
```

### 1. `collect_data.py` - record training data
Form a letter with your hand, then **tap that letter's key** (e.g. press `a`
while signing A). It records a 50-frame burst. Do each letter a few times,
varying distance/angle a little. Press `q` when done. Aim for a few hundred
samples per letter. Data saves to `data/landmarks.csv` (you can re-run to add
more anytime).

### 2. `train_model.py` - train
Trains a Random Forest and prints an accuracy report. Saves the model to
`model/asl_model.joblib`.

### 3. `detect_asl.py` - live detection
Shows the predicted letter in real time with a confidence %. A rolling vote
keeps the letter from flickering. Press `q` to quit.

## Files

| File | Role |
|------|------|
| `common.py` | Shared MediaPipe + feature code (used by collect and detect, so they always match) |
| `collect_data.py` | Step 1 - capture landmark samples to `data/landmarks.csv` |
| `train_model.py` | Step 2 - train the classifier, save to `model/asl_model.joblib` |
| `detect_asl.py` | Step 3 - real-time webcam detection |

## Notes & next steps

- **J and Z** are motion letters in real ASL. This static-handshape model
  captures their start/representative pose only. To handle motion properly,
  you'd classify a short *sequence* of landmark frames (e.g. with a small
  LSTM/temporal model) instead of one frame.
- **More accuracy** = more and more varied training data. Record in the lighting
  and background you'll actually use.
- **Toward the AlphaMini:** keep `common.py` and the trained `.joblib` as-is.
  On the robot, replace the `cv2.VideoCapture(0)` frame source in `detect_asl.py`
  with the AlphaMini's camera feed; the rest of the pipeline is unchanged.
```
