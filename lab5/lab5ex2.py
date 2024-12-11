import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

np.random.seed(42)
X = np.random.rand(2000, 10) * 10
y = np.random.choice([0, 1], size=(2000,), p=[0.9, 0.1]) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print("Performing PCA:")
pca_model = PCA(n_components=2)
X_train_pca = pca_model.fit_transform(X_train_norm)
X_test_pca = pca_model.transform(X_test_norm)

explained_variance = pca_model.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"PCA - Balanced Accuracy (Train): {balanced_accuracy_score(y_train, X_train_pca[:, 0] > np.median(X_train_pca[:, 0])):.2f}")
print(f"PCA - Balanced Accuracy (Test): {balanced_accuracy_score(y_test, X_test_pca[:, 0] > np.median(X_test_pca[:, 0])):.2f}")

print("Performing Kernel PCA:")
kpca_model = KernelPCA(kernel="rbf", gamma=0.01, n_components=2)
X_train_kpca = kpca_model.fit_transform(X_train_norm[:1000])  
X_test_kpca = kpca_model.transform(X_test_norm[:1000])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', alpha=0.6)
ax[0].set_title("PCA Results")
ax[0].set_xlabel("Principal Component 1")
ax[0].set_ylabel("Principal Component 2")

ax[1].scatter(X_train_kpca[:, 0], X_train_kpca[:, 1], c=y_train[:1000], cmap='coolwarm', alpha=0.6)
ax[1].set_title("Kernel PCA Results")
ax[1].set_xlabel("Kernel Component 1")
ax[1].set_ylabel("Kernel Component 2")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label="Individual Variance")
ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where="mid", label="Cumulative Variance")
ax.set_title("Explained Variance by PCA Components")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.legend(loc="best")
plt.show()