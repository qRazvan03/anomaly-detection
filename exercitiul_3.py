import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import zscore

X_train, _, y_train, _ = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

z_scores = np.abs(zscore(X_train))

threshold = np.quantile(z_scores, 0.9)

y_pred = (z_scores > threshold).astype(int)

balanced_acc = balanced_accuracy_score(y_train, y_pred)
print("Acuratețea balansată:", balanced_acc)
