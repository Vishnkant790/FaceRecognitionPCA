# Face Recognition using PCA and ANN

This project implements a face recognition system using Principal Component Analysis (PCA) for feature extraction and an Artificial Neural Network (ANN) for classification.

## Dataset

The face dataset should be placed at:

```text
data/raw/dataset/faces/
```

Each person should have a separate folder:

```text
data/raw/dataset/faces/Aamir/
data/raw/dataset/faces/Ajay/
...
```

Imposter images should be placed at:

```text
data/raw/imposters/
```

The dataset is not uploaded to GitHub because it contains image files.

## Method

1. Load face images from the dataset.
2. Convert each image to grayscale.
3. Resize each image to 100x100.
4. Flatten each image into a vector of 10,000 features.
5. Split data into 60% training and 40% testing.
6. Apply PCA to reduce dimensions.
7. Train ANN classifier using PCA features.
8. Test accuracy for different values of k.
9. Use confidence threshold for imposter detection.

## Results

Best PCA component value:

```text
k = 100
```

Best accuracy:

```text
75%
```

Confidence threshold for imposter detection:

```text
0.90
```

With the current imposter test images, the system predicts them as:

```text
Not enrolled
```

## Accuracy vs k

The graph is saved as:

```text
accuracy_vs_k.png
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Requirements

```text
numpy
opencv-python
matplotlib
scikit-learn
```

## Note

A high confidence threshold helps reject unknown/imposter faces, but it may also reject some enrolled faces with lower confidence.