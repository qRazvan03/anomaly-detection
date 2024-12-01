import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

n_train = int(0.8 * len(X))
X_train_norm, X_test_norm = X_norm[:n_train], X_norm[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print("Fitting OCSVM...")
ocsvm_model = OCSVM(kernel='rbf', gamma='scale', nu=0.1)
ocsvm_model.fit(X_train_norm)

y_test_pred_ocsvm = ocsvm_model.predict(X_test_norm)
roc_auc_ocsvm = roc_auc_score(y_test, y_test_pred_ocsvm)
balanced_acc_ocsvm = balanced_accuracy_score(y_test, y_test_pred_ocsvm)

print(f"OCSVM - Balanced Accuracy: {balanced_acc_ocsvm:.2f}, ROC AUC: {roc_auc_ocsvm:.2f}")

print("Fitting DeepSVDD...")

architectures = [[16, 8], [32, 16, 8]]

for arch in architectures:
    print(f"Testing DeepSVDD architecture {arch}...")
    n_features = X_train_norm.shape[1] 
    deep_svdd_model = DeepSVDD(
        contamination=0.1,
        hidden_neurons=arch,
        batch_size=64,
        epochs=50,
        n_features=n_features
    )
    
    deep_svdd_model.fit(X_train_norm)

    y_test_pred_deepsvdd = deep_svdd_model.predict(X_test_norm)
    roc_auc_deepsvdd = roc_auc_score(y_test, y_test_pred_deepsvdd)
    balanced_acc_deepsvdd = balanced_accuracy_score(y_test, y_test_pred_deepsvdd)

    print(f"DeepSVDD - Balanced Accuracy: {balanced_acc_deepsvdd:.2f}, ROC AUC: {roc_auc_deepsvdd:.2f}")