import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

def load_mammography_dataset():
    data = loadmat("mammography.mat")
    X = data['X']
    y = data['y'].ravel()
    return X, y

class GeneralizedIsolationForest:
    def __init__(self, n_trees=100, subsample_size=256):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = []

    def fit(self, X):
        self.trees = []
        n_samples, n_features = X.shape
        max_depth = int(np.ceil(np.log2(self.subsample_size)))

        for _ in range(self.n_trees):
            subsample_indices = np.random.choice(n_samples, self.subsample_size, replace=False)
            subsample = X[subsample_indices]
            tree = self._build_tree(subsample, depth=0, max_depth=max_depth)
            self.trees.append(tree)

    def _build_tree(self, X, depth, max_depth):
        if depth >= max_depth or X.shape[0] <= 1:
            return {
                "type": "leaf",
                "size": X.shape[0],
            }

        w = np.random.normal(0, 1, X.shape[1])
        w /= np.linalg.norm(w)
        projections = X @ w
        min_proj, max_proj = projections.min(), projections.max()
        split_point = np.random.uniform(min_proj, max_proj)

        left_mask = projections <= split_point
        right_mask = projections > split_point

        return {
            "type": "node",
            "w": w,
            "split_point": split_point,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth),
        }

    def _path_length(self, x, tree, depth=0):
        if tree["type"] == "leaf":
            return depth + self._c(tree["size"])

        projection = x @ tree["w"]
        if projection <= tree["split_point"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    @staticmethod
    def _c(size):
        if size <= 1:
            return 0
        return 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size

    def anomaly_score(self, X):
        scores = []
        for x in X:
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            score = 2 ** (-avg_path_length / self._c(self.subsample_size))
            scores.append(score)
        return np.array(scores)

def main():
    X, y = load_mammography_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    gif = GeneralizedIsolationForest(n_trees=100, subsample_size=256)
    gif.fit(X_train)

    scores = gif.anomaly_score(X_test)

    roc_auc = roc_auc_score(y_test, scores)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(recall, precision)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    main()
