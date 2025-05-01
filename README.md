# Biometrics Project – Facial Feature Classification & Evaluation

This project explores the use of machine learning and biometric performance evaluation in classifying facial features—particularly before and after the addition of facial hair via AI image synthesis. It uses genuine and impostor matching scores to evaluate model robustness across appearance changes.

---

## 📁 Project Structure

1. **`Face_Dataset_conversion.py`**  
   Uses the InstructPix2Pix model to synthetically generate facial hair on clean-shaven images. Saves altered images for downstream analysis.

2. **`final-code.py`**  
   Extracts facial landmarks, calculates pairwise distances, trains classifiers (KNN and SVM in a One-vs-Rest strategy), and computes similarity scores between original and modified images.

3. **`performance.py`**  
   Evaluates system performance using biometric metrics. Generates:
   - ROC Curve
   - DET Curve
   - Score Distribution Plot

---

## 🧪 Requirements

Install all necessary dependencies via:

```bash
pip install -r requirements.txt
```

### `requirements.txt` includes:

- `torch`
- `torchvision`
- `diffusers`
- `pillow`
- `numpy`
- `scikit-learn`
- `matplotlib`

> ⚠️ A CUDA-compatible GPU is **strongly recommended** for faster image processing using the AI model.

---

## 🧠 How to Run the Project

### 1. Convert the Face Dataset

```bash
python Face_Dataset_conversion.py
```

- Loads clean-shaven images from a specified directory
- Applies beard generation using Stable Diffusion's InstructPix2Pix
- Outputs modified images to a target directory

### 2. Train Classifiers & Generate Scores

```bash
python final-code.py
```

- Extracts facial landmark features (68-point)
- Trains KNN and SVM classifiers
- Computes genuine and impostor matching scores

### 3. Evaluate System Performance

```bash
python performance.py
```

- Plots ROC and DET curves
- Visualizes score distributions
- Calculates Equal Error Rate (EER)

---

## 📊 Evaluation Metrics

- **Genuine Score**: Similarity between images of the **same** identity (clean vs. bearded)
- **Impostor Score**: Similarity between images of **different** identities
- **ROC/DET Curves**: Show trade-offs between true/false positives
- **Score Distribution**: Highlights separation between genuine and impostor scores

---

## 📝 Notes

- This project demonstrates the effects of AI-generated modifications on biometric systems.
- You can extend the pipeline to test other facial transformations or models.
- A CUDA-compatible device is preferred for running image generation efficiently.

---

## 📚 Dataset Attribution

All facial images are derived from the **Caltech 1999 Face Dataset**  
📎 [Link to dataset](https://data.caltech.edu/records/6rjah-hdv18)

