# FPV Vision

A production-ready Python project for detecting FPV drone components using YOLO.

---

## Features

- **Train** a YOLO model on a custom FPV part dataset
- **Detect** components from webcam, saved images, or video files
- **Dataset tools** – validate structure, visualise labelled samples, report class balance
- Modular architecture – add new classes without changing code
- CUDA / MPS / CPU auto-detection

---

## Quick start

### 1 · Install

```bash
# Clone the repo
git clone https://github.com/matthewww/YOLO-POC.git
cd YOLO-POC

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install with all dependencies
pip install -e .
```

### 2 · Prepare your dataset

Place your YOLO-format dataset at:

```
data/
  fpv/
    images/
      train/
      val/
    labels/
      train/
      val/
    data.yaml
```

A skeleton `data/fpv/data.yaml` is already provided.

### 3 · Validate dataset

```bash
fpv-vision dataset validate
# or
python scripts/validate_dataset.py
```

### 4 · Visualise labelled samples

```bash
fpv-vision dataset visualize --n 9 --split train
```

### 5 · Train

```bash
fpv-vision train --epochs 100 --batch 16
```

### 6 · Detect

```bash
# Webcam (live)
fpv-vision detect webcam

# Single image
fpv-vision detect image path/to/image.jpg

# Video file
fpv-vision detect video path/to/video.mp4

# Save results
fpv-vision detect image path/to/image.jpg --save --save-dir runs/detect/my_run
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Or edit `configs/default.yaml` directly for full control.

### Classes

Classes are defined in `configs/default.yaml` and `data/fpv/data.yaml`.
Add new classes to both files — no code changes required.

---

## Project structure

```
src/
  fpv_vision/
    config/          # Pydantic settings + YAML defaults
    data/            # Dataset validator & visualiser
    training/        # YOLO training wrapper
    inference/       # Webcam / image / video detection
    visualization/   # Annotation drawing helpers
    utils/           # Device selection, logging
    cli.py           # Typer CLI entry-point

scripts/
  validate_dataset.py   # Standalone dataset validation script

configs/
  default.yaml          # Default configuration

data/
  fpv/
    data.yaml           # YOLO dataset config skeleton
```

---

## FPV component classes

| ID | Class |
|----|-------|
| 0  | motor |
| 1  | flight_controller |
| 2  | esc |
| 3  | camera |
| 4  | vtx |
| 5  | receiver |
| 6  | propeller |
| 7  | lipo_battery |
| 8  | xt60_connector |
| 9  | capacitor |

To add a new class, append it to:
- `configs/default.yaml` → `classes`
- `data/fpv/data.yaml` → `names` (and increment `nc`)

---

## Development

```bash
pip install -e ".[dev]"   # if dev extras are added in future
```

---

## License

MIT
