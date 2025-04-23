#STEP 1: IMPORTING MODULES
# Allows for file to interact with operating system
import os

# Helps find certain file paths (In this case, the .jpg images)
import glob

# Library used to implement math equations
import numpy as np 

# Sklearn classifiers - train_test_split - KNeighborsClassifier - OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

# Computes distances between all pairs of samples from 2 datasets
from sklearn.metrics import pairwise_distances
from performance import Evaluator # External module to graph ROC, DET, and score distribution plots


# STEP 2: Function used to extract 68 facial landmarks from an image using dlib module
def extract_landmarks(image_path):
    import dlib
    import cv2

    #Loading pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Read image
    img = cv2.imread(image_path)
    # If no image found, print warning message
    if img is None:
        print(f"[SKIPPED] Unable to read image: {image_path}")
        return None

    # Convert images to grayscale - 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print(f"[SKIPPED] No face found in: {image_path}")
        return None

    shape = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def extract_features_from_folder(folder_path):
    features = []
    ids = []

    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    print(f"[INFO] Extracting features from: {folder_path} ({len(image_files)} images)")

    for idx, img_path in enumerate(image_files):
        print(f"[{idx + 1}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
        landmarks = extract_landmarks(img_path)
        if landmarks is not None:
            distances = [
                np.linalg.norm(landmarks[i] - landmarks[j])
                for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))
            ]
            features.append(distances)
            try:
                id_number = int(os.path.basename(img_path).split("_")[1].split(".")[0])
                ids.append(id_number)
            except (IndexError, ValueError):
                print(f"[ERROR] Failed to extract ID from: {os.path.basename(img_path)}")

    return np.array(features), np.array(ids)

# === Load Datasets ===

X_clean, y_clean = extract_features_from_folder("Facial_hair_dataset_1/faces")
X_beard, y_beard = extract_features_from_folder("Facial_hair_dataset_1/Faces_Beard_Adjusted")

print(f"[INFO] Clean images processed: {len(X_clean)}")
print(f"[INFO] Bearded images processed: {len(X_beard)}")

# Beard label: 0 = clean-shaven, 1 = bearded
beard_labels = np.array([0]*len(X_clean) + [1]*len(X_beard))
X_all = np.vstack([X_clean, X_beard])
y_ids = np.concatenate([y_clean, y_beard])

# === Run Experiment ===

# Option 1: Train on clean, test on beard
X_train, y_train = X_clean, y_clean
X_test, y_test = X_beard, y_beard

# Build classifier
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
clf.fit(X_train, y_train)

# Similarity scores using pairwise distances (test vs train)
dists = pairwise_distances(X_test, X_train, metric='euclidean')
max_dist = np.max(dists)
similarity_scores = 1 - (dists / max_dist)

# Split genuine/impostor scores
genuine_scores, impostor_scores = [], []
for i in range(len(y_test)):
    for j in range(len(y_train)):
        if y_test[i] == y_train[j]:
            genuine_scores.append(similarity_scores[i, j])
        else:
            impostor_scores.append(similarity_scores[i, j])

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"[INFO] Total genuine pairs: {len(genuine_scores)}")
print(f"[INFO] Total impostor pairs: {len(impostor_scores)}")

# === Evaluate System Performance ===

evaluator = Evaluator(
    num_thresholds=200,
    genuine_scores=genuine_scores,
    impostor_scores=impostor_scores,
    plot_title="Clean → Beard (Cross-Domain Evaluation)"
)

FPR, FNR, TPR = evaluator.compute_rates()
evaluator.plot_score_distribution()
evaluator.plot_det_curve(FPR, FNR)
evaluator.plot_roc_curve(FPR, TPR)
print("Equal Error Rate:", evaluator.get_EER(FPR, FNR))
