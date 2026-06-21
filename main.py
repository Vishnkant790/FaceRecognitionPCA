import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


DATASET_PATH = "data/raw/dataset/faces"
IMPOSTER_PATH = "data/raw/imposters"
IMAGE_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = 0.90
BEST_K = 100


def load_faces(dataset_path):
    face_vectors = []
    labels = []
    label_names = []

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        label_id = len(label_names)
        label_names.append(person_name)

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            image_vector = image.flatten()

            face_vectors.append(image_vector)
            labels.append(label_id)

    return np.array(face_vectors), np.array(labels), label_names


def predict_with_threshold(model, pca_model, image_vector, label_names):
    image_pca = pca_model.transform(image_vector)
    probabilities = model.predict_proba(image_pca)[0]

    max_confidence = np.max(probabilities)
    predicted_label = np.argmax(probabilities)

    if max_confidence < CONFIDENCE_THRESHOLD:
        return "Not enrolled", max_confidence

    return label_names[predicted_label], max_confidence


faces, labels, label_names = load_faces(DATASET_PATH)

print("Dataset summary")
print("Faces shape:", faces.shape)
print("Labels shape:", labels.shape)
print("Label names:", label_names)

X_train, X_test, y_train, y_test = train_test_split(
    faces,
    labels,
    test_size=0.4,
    random_state=42,
    stratify=labels
)

print("\nTrain-test split")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

k_values = [10, 20, 30, 40, 50, 75, 100, 150]
accuracies = []

print("\nAccuracy for different k values")

for k in k_values:
    pca = PCA(n_components=k, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    ann = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=2000,
        random_state=42
    )

    ann.fit(X_train_pca, y_train)
    y_pred = ann.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print("k =", k, "Accuracy =", accuracy)

plt.plot(k_values, accuracies, marker="o")
plt.xlabel("Number of PCA Components (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k value")
plt.grid(True)
plt.savefig("accuracy_vs_k.png", dpi=300)
plt.show()

pca = PCA(n_components=BEST_K, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

ann = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=2000,
    random_state=42
)

ann.fit(X_train_pca, y_train)
y_pred = ann.predict(X_test_pca)
final_accuracy = accuracy_score(y_test, y_pred)

print("\nFinal model")
print("Best k:", BEST_K)
print("Final accuracy:", final_accuracy)
print("Confidence threshold:", CONFIDENCE_THRESHOLD)

print("\nSample test predictions")

for i in range(10):
    image_vector = X_test[i].reshape(1, -1)
    prediction, confidence = predict_with_threshold(
        ann,
        pca,
        image_vector,
        label_names
    )

    print("Actual:", label_names[y_test[i]])
    print("Predicted:", prediction)
    print("Confidence:", confidence)
    print("---")

print("\nTesting imposter images")

if not os.path.exists(IMPOSTER_PATH):
    print("Imposter folder not found:", IMPOSTER_PATH)
else:
    imposter_images = os.listdir(IMPOSTER_PATH)

    if len(imposter_images) == 0:
        print("No imposter images found. Add unknown face images in data/raw/imposters.")
    else:
        for image_name in imposter_images:
            image_path = os.path.join(IMPOSTER_PATH, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            image_vector = image.flatten().reshape(1, -1)

            prediction, confidence = predict_with_threshold(
                ann,
                pca,
                image_vector,
                label_names
            )

            print("Image:", image_name)
            print("Predicted:", prediction)
            print("Confidence:", confidence)
            print("---")