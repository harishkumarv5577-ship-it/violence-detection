# Real-Time Violence Detection in Video

A PyTorch project for training and running a real-time violence detection model on webcam or video input.

The model uses:

- MobileNetV3 as a lightweight frame feature extractor
- GRU temporal classifier over short frame sequences
- Sliding-window real-time inference with probability smoothing

## Project Structure

```text
.
|-- requirements.txt
|-- scripts
|   |-- check_dataset.py
|   |-- detect.py
|   |-- prepare_dataset.py
|   `-- train.py
`-- violence_detection
    |-- __init__.py
    |-- config.py
    |-- dataset.py
    |-- model.py
    |-- realtime.py
    `-- utils.py
```

## 1. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PyTorch installation fails, install the correct build for your system from:

https://pytorch.org/get-started/locally/

## 2. Prepare Dataset

Put videos in this format:

```text
data/
|-- train/
|   |-- violence/
|   |   |-- fight_001.mp4
|   |   `-- ...
|   `-- non_violence/
|       |-- normal_001.mp4
|       `-- ...
`-- val/
    |-- violence/
    `-- non_violence/
```

Recommended public datasets to look up:

- RWF-2000
- Hockey Fight Dataset
- Movies Fight Dataset

For the first test, download a small number of videos into:

```text
data/raw/
|-- violence/
`-- non_violence/
```

Then split them into train and validation folders:

```powershell
python scripts/prepare_dataset.py --raw-dir data/raw --output-dir data --copy
```

Check that OpenCV can read the videos:

```powershell
python scripts/check_dataset.py --data-dir data
```

## 3. Train

```powershell
python scripts/train.py --data-dir data --epochs 10 --batch-size 4
```

The best model will be saved to:

```text
checkpoints/best.pt
```

## 4. Run Real-Time Detection

Webcam:

```powershell
python scripts/detect.py --weights checkpoints/best.pt --source 0
```

Video file:

```powershell
python scripts/detect.py --weights checkpoints/best.pt --source path\to\video.mp4
```

Press `q` to stop.

## 5. Run H2 Web App

The `webapp` folder adds an H2 database for storing users who sign up with Gmail, full name, password, and their video detection results.

```powershell
python webapp/tools/download_h2.py
powershell -ExecutionPolicy Bypass -File webapp/run.ps1
```

Open:

```text
http://localhost:8080
```

The web app now uses separate professional pages:

```text
/login      - Gmail/password signup and login page
/dashboard  - upload video or webcam selection page
/output     - final detection result page
```

Upload flow:

```text
Login -> Dashboard -> Upload Video -> ML Processing -> Output Page -> H2 History
```

The H2 database file is stored at:

```text
webapp/database/violence_detection.mv.db
```

After a user is saved in the web app, detection results can also be stored from Python:

```powershell
python scripts/detect.py --weights checkpoints/best.pt --source data/val/violence/airtlab_cam1_101.mp4 --sequence-length 8 --no-display --output outputs/test.mp4 --api-url http://localhost:8080 --email student@gmail.com
```

The web app stores:

- Gmail/email
- Full name
- Password hash
- Video name
- Detection result
- Confidence score
- Output video path
- Processed time

## Important Notes

This project is a starting point for academic/demo use. Real-world violence detection can produce false positives and should not be used for punitive or high-stakes decisions without human review, careful validation, and bias testing.
