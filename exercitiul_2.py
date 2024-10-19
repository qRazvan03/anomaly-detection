from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, balanced_accuracy_score

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

model = KNN(contamination=0.1)
model.fit(X_train)

y_train_pred = model.labels_  # Predicții pentru antrenament
y_test_pred = model.predict(X_test)  # Predicții pentru testare

cm = confusion_matrix(y_test, y_test_pred)
balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

print("Matricea de confuzie:\n", cm)
print("Acuratețea balansată:", balanced_acc)

fpr, tpr, _ = roc_curve(y_test, y_test_pred)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()