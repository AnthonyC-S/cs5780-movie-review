from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import joblib

#np.set_printoptions(threshold=sys.maxsize)

# Given code from Kaggle competition
def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}

train_data = load_npz('train.npz')
test_data = load_npz('test.npz')

# Extract feature embeddings, labels, and uid
train_emb1, train_emb2, train_labels = train_data['emb1'], train_data['emb2'], train_data['preference']
test_emb1, test_emb2, test_uid = test_data['emb1'], test_data['emb2'], test_data['uid']

# Combine training embeddings
all_train_embeddings = np.concatenate((train_emb1, train_emb2), axis=0)

# Combine training labels
neg_train_labels = np.logical_not(train_labels).astype(np.int8)
all_train_labels = np.concatenate((train_labels, neg_train_labels))

# Shuffle and split data. Note, need to see if different splits (eg 10%, 15%) affect performance
xTr, xVal, yTr, yVal = train_test_split(all_train_embeddings, all_train_labels, test_size=0.2, random_state=42)


# # Hyperparameter Tuning for SVM Classifier
# svm_classifier = SVC()

# # Parameters to try in Grid/Randomized Search. Note, need to do more searching slighly higher.
# params = {'C': [0.5, 1, 1.5],
#               'gamma': [0.5, 1, 1.5],
#               'kernel': ['rbf']}

# # Parameter Grid Search
# grid = GridSearchCV(svm_classifier, params, cv=2, refit=True, verbose=3)
# grid.fit(xTr, yTr)
# print(grid.best_params_)
# print(grid.best_estimator_)

# # SVM Classifier Model, C=1.5, gamma=3
# svm_classifier = SVC(kernel='rbf', C=1.5, gamma=3, probability=True, verbose=1)
# svm_classifier.fit(xTr, yTr)
# # Accuracy: 0.8397333333333333 C=1.5, gamma=1.5
# # Accuracy: 0.8401333333333333 C=1.5, gamma=2

# # Save SVM Classifier Model
# joblib.dump(svm_classifier, "svm_classifier_c=1_5_gamma=3.pkl")

# # Evaluate accuracy
# predictions = svm_classifier.predict(xVal)
# accuracy = accuracy_score(yVal, predictions)
# print(f"Accuracy: {accuracy}")

# Load Data
svm_classifier = joblib.load("svm_classifier_c=1_5_gamma=3.pkl")

# Get test probabilities for emb1 and emb2
test_emb1_probs = svm_classifier.predict_proba(test_emb1)
test_emb2_probs = svm_classifier.predict_proba(test_emb2)

# Create array to store preditions on test
test_predictions = np.empty([len(test_emb1)], dtype=np.int8)

# Add predictions to array
for i in range(len(test_emb1_probs)):
    if (test_emb1_probs[i])[0] > (test_emb2_probs[i])[0]:
        test_predictions[i] = 1
    elif (test_emb1_probs[i])[0] < (test_emb2_probs[i])[0]:
        test_predictions[i] = 0
    else:
        test_predictions[i] = 1 if (test_emb1_probs[i])[0] > (test_emb1_probs[i])[1] else 0


# Sanity check to make sure predictions match probabilities
# for i in range(50):
#     print(test_emb1_probs[i], test_emb2_probs[i], test_predictions[i])

# Apparently I had my 0 and 1 mixed up in test predictions and got a score of 0.07
# Inverting the predictions below brings score up to 0.92 on leaderboard
# Figured out why, in predict_proba, the returned estimates for all classes are ordered by the label of classes.
test_predictions = np.logical_not(test_predictions).astype(np.int8)

uid_preds = np.vstack((test_uid, test_predictions)).T

# Save to csv, note need to manually add header
np.savetxt("submission_2_svm.csv", uid_preds, delimiter=",", fmt='%s')