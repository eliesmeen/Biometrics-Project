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
import matplotlib.pyplot as plt


# STEP 2: Function used to extract 68 facial landmarks from an image using dlib module
def extract_landmarks(image_path):
    import dlib
    import cv2

    #Loading pre-trained face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Read image from file path
    img = cv2.imread(image_path)
    # If no image found, print warning message
    if img is None:
        print(f"[SKIPPED] Unable to read image: {image_path}")
        return None

    # Convert images to grayscale - Face detector requires this
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # If no face is detected, print a warning message
    if len(faces) == 0:
        print(f"[SKIPPED] No face found in: {image_path}")
        return None

    # Uses the first face detected to predict the 68 landmarks
    shape = predictor(gray, faces[0])
    #Converts the dlib shape object into x and y coordinates
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    #returns (x, y) coordinates for further processing
    return landmarks

# STEP 3: Function to extract features - pairwise distances between facial landmarks - from all images in a folder
def extract_features_from_folder(folder_path):

    # Empty lists to store extracted features
    features = []
    ids = []

    # Get sorted list of all .jpg images
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    print(f"[INFO] Extracting features from: {folder_path} ({len(image_files)} images)")

    # Iterate through each image within the folder
    for idx, img_path in enumerate(image_files):
        # Prints progresss message while running
        print(f"[{idx + 1}/{len(image_files)}] Processing: {os.path.basename(img_path)}")
        #Extarcts 68 landmarks from image
        landmarks = extract_landmarks(img_path)

        # Processed only if landmark is successfully extracted
        if landmarks is not None:
            #Computes pairwise Euclidean distances between landmark points
            distances = [
                np.linalg.norm(landmarks[i] - landmarks[j])
                for i in range(len(landmarks)) for j in range(i + 1, len(landmarks))
            ]
            # Adds computed distance to feature vector for the feature list
            features.append(distances)
            # Gets ID number from the image name file
            try:
                id_number = int(os.path.basename(img_path).split("_")[1].split(".")[0])
                ids.append(id_number)
            # Error message if ID could not be extracted
            except (IndexError, ValueError):
                print(f"[ERROR] Failed to extract ID from: {os.path.basename(img_path)}")

    # Conversion of the list of features and ids into arrays
    return np.array(features), np.array(ids)


# STEP 4: Load Datasets and call function extract_features_from_folder - The clean shaven dataset and facial hair dataset is extarcted seperatately
# X_ value stores the feature vectors 
# Y_ value stores the ID Number
X_clean, y_clean = extract_features_from_folder("Facial_hair_dataset_1/faces")
X_beard, y_beard = extract_features_from_folder("Facial_hair_dataset_1/Facial_Hair_Thick_beard")

#Prints success status of Clean and Bearded images
print(f"[INFO] Clean images processed: {len(X_clean)}")
print(f"[INFO] Bearded images processed: {len(X_beard)}")

# Beard label: 0 = clean-shaven, 1 = bearded - used for grouping
beard_labels = np.array([0]*len(X_clean) + [1]*len(X_beard))

#Combines bearded and clean datasets into single dataset
X_all = np.vstack([X_clean, X_beard])
y_ids = np.concatenate([y_clean, y_beard])

# === Experiment 1: Train on clean, test on beard ===
print("\n[EXPERIMENT 1] Train on clean -> Test on beard")

X_train, y_train = X_clean, y_clean
X_test, y_test = X_beard, y_beard

clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
clf.fit(X_train, y_train)

dists = pairwise_distances(X_test, X_train, metric='euclidean')
max_dist = np.max(dists)
similarity_scores = 1 - (dists / max_dist)

genuine_scores = []
impostor_scores = []

for i in range(len(y_test)):
    for j in range(len(y_train)):
        if y_test[i] == y_train[j]:
            genuine_scores.append(similarity_scores[i, j])
        else:
            impostor_scores.append(similarity_scores[i, j])

evaluator = Evaluator(
    num_thresholds=200,
    genuine_scores=np.array(genuine_scores),
    impostor_scores=np.array(impostor_scores),
    plot_title="Clean to Beard (Cross-Domain Evaluation)"
)

FPR, FNR, TPR = evaluator.compute_rates()
evaluator.plot_score_distribution()
evaluator.plot_det_curve(FPR, FNR)
evaluator.plot_roc_curve(FPR, TPR)
print("Equal Error Rate (Clean to Beard):", evaluator.get_EER(FPR, FNR))


# === Experiment 2: Train on beard, test on clean ===
print("\n[EXPERIMENT 2] Train on beard -> Test on clean")

X_train, y_train = X_beard, y_beard
X_test, y_test = X_clean, y_clean

clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
clf.fit(X_train, y_train)

dists = pairwise_distances(X_test, X_train, metric='euclidean')
max_dist = np.max(dists)
similarity_scores = 1 - (dists / max_dist)

genuine_scores = []
impostor_scores = []

for i in range(len(y_test)):
    for j in range(len(y_train)):
        if y_test[i] == y_train[j]:
            genuine_scores.append(similarity_scores[i, j])
        else:
            impostor_scores.append(similarity_scores[i, j])

evaluator = Evaluator(
    num_thresholds=200,
    genuine_scores=np.array(genuine_scores),
    impostor_scores=np.array(impostor_scores),
    plot_title="Beard to Clean (Cross-Domain Evaluation)"
)

FPR, FNR, TPR = evaluator.compute_rates()
evaluator.plot_score_distribution()
evaluator.plot_det_curve(FPR, FNR)
evaluator.plot_roc_curve(FPR, TPR)
print("Equal Error Rate (Beard to Clean):", evaluator.get_EER(FPR, FNR))

