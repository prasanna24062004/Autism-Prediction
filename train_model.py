import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

data_dir = "dataset"
categories = ["autistic", "non_autistic"]

data = []
labels = []

for label, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))  # HOG prefers square sizes
        features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                          orientations=9, visualize=True)
        data.append(features)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "autism_svm_model.pkl")
