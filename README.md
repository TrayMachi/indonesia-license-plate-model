# Indonesian License Plate Model

Colab-first training and Android export workspace for detecting Indonesian license plates.

## Repository layout

```text
indonesia-license-plate-model/
├── notebooks/
│   ├── 01_dataset.ipynb          # Validate and inspect a YOLO dataset
│   ├── 02_train_detector.ipynb   # Train on a Colab GPU
│   ├── 03_evaluate.ipynb         # Evaluate a saved checkpoint
│   └── 04_export_android.ipynb   # Export a trained model to LiteRT/TFLite
├── configs/
│   └── dataset.yaml
├── src/
│   ├── preprocess.py
│   └── postprocess.py
├── requirements.txt
└── .gitignore
```

## Dataset format

The notebooks use detector-compatible YOLO labels and a dataset stored in Google Drive:

```text
MyDrive/indonesia-license-plate-model/dataset_yolo/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Each prepared label file contains one normalized row per plate:

```text
class_id x_center y_center width height
```

The source Kaggle detection labels may include a sixth plate-text field after the bounding
box. `01_dataset.ipynb` removes that optional field when creating `dataset_yolo`; the
original `dataset/` folder is left untouched if it already exists. If a previous run left
`dataset_yolo/` incomplete, the notebook detects the missing split, backs up that folder,
and rebuilds a complete dataset.

The default configuration has one class, `plate`. Update `configs/dataset.yaml` if the
Drive folder or class list differs.

## Colab workflow

1. Open `notebooks/01_dataset.ipynb` from GitHub in Google Colab. It downloads the Kaggle
   archive, prepares the detection subset, strips any optional plate-text field, creates a
   deterministic 80/10/10 split, and validates the data.
2. Run `notebooks/02_train_detector.ipynb` with a GPU runtime. Checkpoints and metrics are
   saved under `MyDrive/indonesia-license-plate-model/runs/`.
3. Use `notebooks/03_evaluate.ipynb` to compare validation metrics and inspect predictions.
4. Run `notebooks/04_export_android.ipynb` to export `best.pt` as a `.tflite` model.

The training notebook clones this repository when needed, mounts Drive, checks the GPU,
installs the project dependencies, validates the dataset, and saves the best checkpoint to
Drive.

## Android integration

`src/preprocess.py` contains model-input letterboxing helpers. `src/postprocess.py` contains
format-agnostic box conversion, scaling, and class-wise NMS helpers for the Android inference
layer. For the current YOLO11n LiteRT export, `to_model_input` returns RGB float32 input in
`[1, 3, 640, 640]` NCHW layout. The model emits raw `[1, 5, 8400]` detections because NMS is
disabled; `decode_yolo11_output` transposes and decodes that output before NMS. Apply
`scale_boxes_to_original` after decoding to undo the letterbox padding.

## Local setup

```bash
python -m pip install -r requirements.txt
```

This project intentionally keeps images, labels, checkpoints, and exported binaries out of
Git; use Google Drive for those artifacts.

## Dataset source

The initial detection dataset is [Indonesian License Plate Dataset on Kaggle](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset),
version 1. It contains 1,000 full images with YOLO detection labels and a separate cropped
recognition subset. Kaggle currently lists the license as Unknown, so confirm permission with
the uploader before commercial use or redistribution.
