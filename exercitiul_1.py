from pyod.utils.data import generate_data
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.title('Set de date de antrenament cu outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

