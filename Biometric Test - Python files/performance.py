import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, num_thresholds, genuine_scores, impostor_scores, plot_title, epsilon=1e-12):
        """
        Initialize the Evaluator object.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(0, 1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.
        """
        x = np.mean(self.genuine_scores) - np.mean(self.impostor_scores)
        y = np.sqrt(0.5 * (np.std(self.genuine_scores)**2 + np.std(self.impostor_scores)**2))
        return x / (y + self.epsilon)

    
    def plot_score_distribution(self):
        plt.figure(figsize=(10, 6))
    
    
        plt.hist(self.genuine_scores, bins=30, alpha=0.7, color='green', 
             density=True, label='Genuine')
        plt.hist(self.impostor_scores, bins=30, alpha=0.7, color='red', 
             density=True, label='Impostor')
    
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title(f'Score Distribution\n(d-prime = {self.get_dprime():.2f})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 1)  
        plt.tight_layout()
        plt.savefig('score_distribution.png', dpi=300) # if this doesn't work I will provide png in zip 
        plt.show()

    def compute_rates(self):
        """
        Compute False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR).
        """
        fpr, fnr, tpr = [], [], []
        for threshold in self.thresholds:
            fp = np.sum(self.impostor_scores >= threshold)
            fn = np.sum(self.genuine_scores < threshold)
            tp = np.sum(self.genuine_scores >= threshold)
            tn = np.sum(self.impostor_scores < threshold)
            
            fpr.append(fp / (fp + tn + self.epsilon))
            fnr.append(fn / (fn + tp + self.epsilon))
            tpr.append(tp / (tp + fn + self.epsilon))
        return np.array(fpr), np.array(fnr), np.array(tpr)

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
        """
        diff = np.abs(FPR - FNR)
        idx = np.argmin(diff)
        return (FPR[idx] + FNR[idx]) / 2

    def plot_det_curve(self, FPR, FNR):
        plt.figure(figsize=(8, 8))
    
    
        plt.plot(FPR, FNR, label='DET Curve', linewidth=2)
        plt.scatter([self.get_EER(FPR, FNR)], [self.get_EER(FPR, FNR)], 
                color='red', s=100, label=f'EER: {self.get_EER(FPR, FNR):.2f}')
    
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('False Negative Rate', fontsize=12)
        plt.title('Detection Error Tradeoff (DET) Curve', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('det_curve.png', dpi=300)
        plt.show()

    def plot_roc_curve(self, FPR, TPR):
        plt.figure(figsize=(8, 8))
    
        plt.plot(FPR, TPR, label='ROC Curve', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300)
        plt.show()
