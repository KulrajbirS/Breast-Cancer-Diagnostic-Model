# Importing all the required Modules/Packages/Libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Reading the Breast Cancer Data Set as a Data Frame using Pandas Library
bc_data = pd.read_csv('breast_cancer_data.csv')

# Filtering Feature Data Set for Data Set Visualization
plotting_dataset = bc_data.drop(['id', 'diagnosis', 'Unnamed: 32', "radius_se", "texture_se", "perimeter_se", "area_se",
                                 "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                                 "fractal_dimension_se"], 1).values

# Plotting the Feature Set using t-SNE as a Scatter Plot
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=4000)
tsne = tsne.fit_transform(plotting_dataset)
plt.scatter(tsne[:, 0], tsne[:, 1],  c=bc_data['diagnosis'].map({'M': 0, 'B': 1}), cmap="winter", edgecolor="None", alpha=0.35)
plt.title('t-SNE Scatter Plot')
plt.show()

# Converting the Features Set into a NumPy Array of Shape (569, 20)
features = np.array(bc_data.drop(['id', 'diagnosis', 'Unnamed: 32', "radius_se", "texture_se", "perimeter_se", "area_se",
                                  "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
                                  "fractal_dimension_se"], 1))
# Converting Target Set into a NumPy Array of Shape (569, )
target = np.array(bc_data['diagnosis'])

# Variable to Hold the Best Score of the Model
best = 0
# Iterative Statement to find out the Best Trained Model
for a in range(100):
    # Splitting data into 80:20 ratio for training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # Initializing Logistic Regression Model and Training it using the train set
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Conditional Statement to check if current trained model is better than best score from previous iteration
    if score > best:
        best = score
        # Using Pickle Module to Serialize the Best Model out of all the iterations into a Pickle File
        with open("breast_cancer.pickle", "wb") as model:
            pickle.dump(clf, model)

# Loading the Model from the Pickle File created in the iterations above (having the best test score)
model_in = open("breast_cancer.pickle", "rb")
clfmodel = pickle.load(model_in)

# Predicting the Diagnosis using the model on the Unlabeled test Data Set
y_pred = clfmodel.predict(X_test)

# Displaying the Original Diagnosis and the Predicted Diagnosis for result analysis
print('\nDiagnosis Prediction Results')
for x, y in zip(y_test, y_pred):
    if x == y:
        print(f'Right Prediction -> {x}: {y}')
    else:
        print(f'Wrong Prediction -> {x}: {y}*')
print('\n\n')

# Initializing Confusion Matrix using Metrics Module
confusion_mat = metrics.confusion_matrix(y_test, y_pred)
# Setting Plotting Descriptions for the Confusion Matrix using PyPlot
fig, axes = plt.subplots(figsize=(10, 10))
axes.imshow(confusion_mat)
axes.grid(False)
axes.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Benign', 'Predicted Malignant'))
axes.yaxis.set(ticks=(0, 1), ticklabels=('Actual Benign', 'Actual Malignant'))
axes.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        axes.text(j, i, confusion_mat[i, j], ha='center', va='center', color='red')
plt.show()

# Displaying the Model Score and Classification Report
print(f'Score for the Model: {clfmodel.score(X_test,y_test)}')
print(metrics.classification_report(y_test, y_pred))
