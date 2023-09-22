# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Loading the Iris dataset
print("Loading the Iris dataset...")
data = pd.read_csv('iris_dataset.csv')

# Handling any missing values in the dataset
print("Handling missing values...")
data.dropna(subset=['species'], inplace=True)

# Separating features (X) and the target variable (y)
print("Separating features and the target variable...")
X = data.drop(columns=['species'])
y = data['species']

# Splitting the dataset into training and testing sets (80% train, 20% test)
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initializing and training the K-Nearest Neighbors classifier
print("Initializing and training the classifier...")
knn = KNeighborsClassifier(n_neighbors=3)  # Using 3 neighbors for classification
knn.fit(X_train, y_train)

# Making predictions on the test set
print("Making predictions on the test set...")
y_pred = knn.predict(X_test)

# Evaluating the model
print("Evaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Printing the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plotting ROC curves for each class
print("Plotting ROC curves...")
class_labels = knn.classes_
roc_curves = {}

for i, class_label in enumerate(class_labels):
    y_true_binary = (y_test == class_label).astype(int)
    y_prob = knn.predict_proba(X_test)[:, i]
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    auc = roc_auc_score(y_true_binary, y_prob)
    roc_curves[class_label] = (fpr, tpr, auc)

plt.figure(figsize=(8, 6))
for class_label, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
plt.legend(loc='lower right')
plt.show()

# Conclusion
print("The K-Nearest Neighbors model has been successfully trained and evaluated for Iris flower classification.")
print("It can accurately classify Iris flowers into different species based on their sepal and petal measurements.")
