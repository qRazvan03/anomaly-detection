import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import zscore

X_train, _, y_train, _ = generate_data(n_train=1000, n_test=0, n_features=3, contamination=0.1)

z_scores = np.abs(zscore(X_train, axis=0))

threshold = np.quantile(np.mean(z_scores, axis=1), 0.9)

y_pred = (np.mean(z_scores, axis=1) > threshold).astype(int)

balanced_acc = balanced_accuracy_score(y_train, y_pred)
print("Acuratețea balansată:", balanced_acc)
