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

The notebooks expect YOLO detection labels and a dataset stored in Google Drive:

```text
MyDrive/indonesia-license-plate-model/dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Each label file contains one normalized row per plate:

```text
class_id x_center y_center width height
```

The default configuration has one class, `plate`. Update `configs/dataset.yaml` if the
Drive folder or class list differs.

## Colab workflow

1. Open `notebooks/01_dataset.ipynb` from GitHub in Google Colab and validate the data.
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
layer. Confirm the exported model's input/output shapes in the export notebook before wiring
the final tensor adapter into the app.

## Local setup

```bash
python -m pip install -r requirements.txt
```

This project intentionally keeps images, labels, checkpoints, and exported binaries out of
Git; use Google Drive for those artifacts.
