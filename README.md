Biometrics Project – Facial Feature Classification & Evaluation
This project applies machine learning and biometric evaluation techniques to a dataset of facial images. The objective is to classify individuals based on modified facial features (such as beard generation using AI) and evaluate the performance of the classifiers using genuine and impostor matching scores.

Please install required modules:
pip install -r requirements.txt

Project Structure
1) Face_Dataset_conversion.py
Uses a pre-trained AI model (InstructPix2Pix) to synthetically modify facial images (e.g., by adding a beard), creating an altered version of the dataset.

2) final-code.py
Extracts image features, trains classifiers (KNN and SVM), and calculates similarity scores between original and altered images.

3) performance.py
Evaluates the classification system using biometric performance metrics such as ROC and DET curves and score distribution plots.

Requirements
Make sure the following Python packages are installed:

torch

torchvision

diffusers

pillow

numpy

scikit-learn

matplotlib

A CUDA-compatible GPU is recommended for faster image generation.

How to Run the Project
1. Convert the Face Dataset
Run Face_Dataset_conversion.py to:

Load images from a specified directory

Apply beard generation using the InstructPix2Pix model

Save the modified dataset for further analysis

2. Train Models and Generate Scores
Run final-code.py to:

Load original and altered image features

Train KNN and SVM classifiers with One-vs-Rest classification

Compute similarity scores between image pairs

3. Evaluate the System
Run performance.py to:

Analyze genuine and impostor score distributions

Generate ROC and DET plots

Assess the system's classification effectiveness across thresholds

Evaluation Metrics
Genuine Score: Measures similarity between images of the same person (before vs. after beard generation)

Impostor Score: Measures similarity between different individuals

ROC/DET Curves: Visual representations of true positive vs. false positive rates

Score Distribution: Shows separation between genuine and impostor scores

Notes
This project demonstrates how biometric systems can be tested under adversarial or synthetic conditions using AI-generated facial modifications. The codebase can be extended to include additional facial transformations, alternative classification models, or larger datasets.

All facial images are credited to the Caltech 1999 Dataset.
The dataset can be accessed here: https://data.caltech.edu/records/6rjah-hdv18
