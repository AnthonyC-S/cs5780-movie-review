"""\
CS 4780/5780, Fall 2023
Kaggle Competition on Movie Review Preference Analysis
https://www.kaggle.com/competitions/movie-review-preference-analysis/


This script uses a SVM to train and classify the dataset after hyperparameter tuning via GridSearch.
Further improved performance by including very high confidence prediction features into training data,
and then retraining.
For more in-depth summary, please see README.txt

Authors: Anthony Coffin-Schmitt (awc93)
         Hansal Shah (hms262)
"""

# Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Function to load from npz, given code from Kaggle competition
def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}

# Load training and testing data
train_data = load_npz('train.npz')
test_data = load_npz('test.npz')

# Extract feature embeddings, labels, and uid from training and testing data
train_emb1, train_emb2, train_labels = train_data['emb1'], train_data['emb2'], train_data['preference']
test_emb1, test_emb2, test_uid = test_data['emb1'], test_data['emb2'], test_data['uid']

# Concat training embeddings, doubles the training size to 37,500
all_train_embeddings = np.concatenate((train_emb1, train_emb2), axis=0)

# Invert training labels and concat all lables to match train_emb1 + train_emb2
neg_train_labels = np.logical_not(train_labels).astype(np.int8)
all_train_labels = np.concatenate((train_labels, neg_train_labels))

# Shuffle and split data for parameter search:
xTr, xVal, yTr, yVal = train_test_split(all_train_embeddings, all_train_labels, test_size=0.2, random_state=42)

# Function to evaluate trained model accuracy, uses yVal to test
def get_accuracy(svm_classifier):
    preds = svm_classifier.predict(xVal)
    accuracy = accuracy_score(yVal, preds)
    print(f"Accuracy: {accuracy}")

# Vanilla SVM Model with default params, C=1, gamma=1, no CV, to get idea of initial accuracy
svm_classifier_vanilla = SVC(kernel='rbf', verbose=1)
svm_classifier_vanilla.fit(xTr, yTr)
get_accuracy(svm_classifier_vanilla)
# Accuracy: 0.83293


################################################################################################
# Hyperparameter Tuning

# Function to perform GridSearchCV with given parameters, cross validiation set to 3
def perform_grid_search(params):
    svm_classifier = SVC()
    grid = GridSearchCV(svm_classifier, params, scoring='accuracy', cv=3, refit=True, verbose=3)
    grid.fit(all_train_embeddings, all_train_labels)
    print(f"Best Params: {grid.best_params_}")
    print(f"Best Estimator: {grid.best_estimator_}")

# Initial Parameter GridSearchCV
params_1 = {'C': [0.5, 1, 1.5],
            'gamma': [0.5, 1, 1.5, 3],
            'kernel': ['rbf']}
perform_grid_search(params_1)
# Best Performing Params for params_1:
# {'C': 1.5, 'gamma': 3, 'kernel': 'rbf'}
# SVC(C=1.5, gamma=3)


# Parameter GridSearchCV 2 . Note this will likely run overnight
params_2 = {'C': [1.25, 1.5, 1.75, 2, 2.25, 2.5],
              'gamma': [2.25, 2.5, 2.75, 3, 3.25, 3.5],
              'kernel': ['rbf']}
perform_grid_search(params_2)
# Best Performing Params for params_2:
# Results: Best Param = SVC(C=2, gamma=2.5)


# Function to generate list of uid, predictions of given model. Note, model must have been 
# trained with predictions as it will use the higher confidence prediction in cases reviews match
def generate_preds_lst(svm_classifier):
    # Get test probabilities for emb1 and emb2
    test_emb1_probs = svm_classifier.predict_proba(test_emb1)
    test_emb2_probs = svm_classifier.predict_proba(test_emb2)

    # Create array to store preditions on test
    test_predictions = np.empty([len(test_emb1)], dtype=np.int8)

    # Add predictions to array - if emb1 and emb2 both predict the same review class, go with
    # the prediction that has higher confidence
    for i in range(len(test_emb1_probs)):
        if (test_emb1_probs[i])[0] > (test_emb2_probs[i])[0]:
            test_predictions[i] = 0
        elif (test_emb1_probs[i])[0] < (test_emb2_probs[i])[0]:
            test_predictions[i] = 1
        else:
            test_predictions[i] = 0 if (test_emb1_probs[i])[0] > (test_emb1_probs[i])[1] else 1

    # Combine uid and predictions into array before saving to csv
    uid_preds = np.vstack((test_uid, test_predictions)).T

    return uid_preds

 # Sanity check to make sure predictions match probabilities
def pred_sanity_check(test_emb1_probs, test_emb2_probs):
    for i in range(25):
        print(test_emb1_probs[i], test_emb2_probs[i], test_emb2_probs[i])


################################################################################################
# Kaggle Submission 2 - Score: 0.92067

# Init SVM Model (C=1.5, gamma=3) for Submission 1 and Submission 2 to Kaggle:
# svm_classifier_c1_5_gamma3 = SVC(kernel='rbf', C=1.5, gamma=3, probability=True, verbose=1)

# Train SVM Model (C=1.5, gamma=3)
# svm_classifier_c1_5_gamma3.fit(all_train_embeddings, all_train_labels)

# Save SVM Model (C=1.5, gamma=3)
# joblib.dump(svm_classifier_c1_5_gamma3, "svm_classifier_c=1_5_gamma=3.pkl")

# Load SVM Model (C=1.5, gamma=3)
# svm_classifier = joblib.load("svm_classifier_c=1_5_gamma=3.pkl")

# Get SVM Model (C=1.5, gamma=3) prediction list
# uid_preds = generate_preds_lst(svm_classifier)

# Create SVM Model (C=1.5, gamma=3) CSV file for Submission 2
# np.savetxt('submission_2_svm.csv', uid_preds, delimiter=',', header='uid,preference', comments='', fmt='%s')


################################################################################################
# Kaggle Submission 3 - Score: 0.92197

# Init SVM Model (C=2.5, gamma=2.25) for Submission 1 and Submission 3 to Kaggle:
# svm_classifier_c2_5_gamma2_25 = SVC(kernel='rbf', C=2.5, gamma=2.25, probability=True, verbose=1)

# Train SVM Model (C=2.5, gamma=2.25)
# svm_classifier_c2_5_gamma2_25.fit(all_train_embeddings, all_train_labels)

# Save SVM Model (C=2.5, gamma=2.25)
# joblib.dump(svm_classifier_c2_5_gamma2_25, "svm_classifier_c=2_5_gamma=2_25.pkl")

# Load SVM Model (C=2.5, gamma=2.25)
# svm_classifier_c2_5_gamma2_25 = joblib.load("svm_classifier_c=2_5_gamma=2_25.pkl")

# Get SVM Model (C=2.5, gamma=2.25) prediction list
# uid_preds = generate_preds_lst(svm_classifier_c2_5_gamma2_25)

# Create SVM Model (C=2.5, gamma=2.25) CSV file for Submission 3
# np.savetxt('submission_3_svm.csv', uid_preds, delimiter=',', header='uid,preference', comments='', fmt='%s')


################################################################################################
# Kaggle Submission 4 - Score: 0.9275

# Load last SVM Model (C=2.5, gamma=2.25)
svm_classifier_c2_5_gamma2_25 = joblib.load("svm_classifier_c=2_5_gamma=2_25.pkl")

# Get test prediction probabilities for SVM Model (C=2.5, gamma=2.25)
test_emb1_probs = svm_classifier_c2_5_gamma2_25.predict_proba(test_emb1)
test_emb2_probs = svm_classifier_c2_5_gamma2_25.predict_proba(test_emb2)

# Get indices of test data with high confidence (hc) predictions, based on SVM Model (C=2.5, gamma=2.25)
# Both review predictions > 0.9 and reviews do not match
hc_test_emb_filter = []
hc_test_emb_1_label = np.empty((0), dtype=np.int8)
hc_test_emb_2_label = np.empty((0), dtype=np.int8)

for i in range(len(test_emb1)):
    if ((test_emb1_probs[i])[0] > 0.9 and (test_emb2_probs[i])[1] > 0.9):
        hc_test_emb_filter.append(i)
        hc_test_emb_1_label = np.append(hc_test_emb_1_label, 0)
        hc_test_emb_2_label = np.append(hc_test_emb_2_label, 1)

    if ((test_emb1_probs[i])[1] > 0.9 and (test_emb2_probs[i])[0] > 0.9):
        hc_test_emb_filter.append(i)
        hc_test_emb_1_label = np.append(hc_test_emb_1_label, 1)
        hc_test_emb_2_label = np.append(hc_test_emb_2_label, 0)

# Filter out test dataset based on hc filter
hc_test_emb1 = test_emb1[hc_test_emb_filter]
hc_test_emb2 = test_emb2[hc_test_emb_filter]

# Add the high confidence test predictions to training datasets
all_train_embeddings = np.concatenate((all_train_embeddings, hc_test_emb1, hc_test_emb2), axis=0)
all_train_labels = np.concatenate((all_train_labels, hc_test_emb_1_label, hc_test_emb_2_label), axis=0)

# Retrain Model HC_SMV (C=2.5, gamma=2.25) with test features added to training
# Init SVM Model (C=2.5, gamma=2.25) for Submission 1 and Submission 3 to Kaggle:
hc_svm_classifier_c2_5_gamma2_25 = SVC(kernel='rbf', C=2.5, gamma=2.25, probability=True, verbose=1)

# Train SVM Model (C=2.5, gamma=2.25)
hc_svm_classifier_c2_5_gamma2_25.fit(all_train_embeddings, all_train_labels)

# Save SVM Model (C=2.5, gamma=2.25)
joblib.dump(hc_svm_classifier_c2_5_gamma2_25, "hc_svm_classifier_c=2_5_gamma=2_25.pkl")

# Load SVM Model (C=2.5, gamma=2.25)
hc_svm_classifier_c2_5_gamma2_25 = joblib.load("hc_svm_classifier_c=2_5_gamma=2_25.pkl")

# Get SVM Model (C=2.5, gamma=2.25) prediction list
uid_preds = generate_preds_lst(hc_svm_classifier_c2_5_gamma2_25)

# Create SVM Model (C=2.5, gamma=2.25) CSV file for Submission 3
np.savetxt('submission_4_hc_svm_COMPARE.csv', uid_preds, delimiter=',', header='uid,preference', comments='', fmt='%s')