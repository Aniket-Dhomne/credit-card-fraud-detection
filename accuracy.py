from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
import streamlit as st

# load data
creditcard_df = pd.read_csv('creditcard.csv')


#in-balance dataset(more fradulent data)
creditcard_df['Class'].value_counts()

#balancing the dataset
legit=creditcard_df[creditcard_df.Class==0]
fraud=creditcard_df[creditcard_df.Class==1]

legit_bal=legit.sample(n=len(fraud))
creditcard_df=pd.concat([legit_bal,fraud],axis=0)

creditcard_df.groupby('Class').mean()

X=creditcard_df.drop('Class',axis=1)
y=creditcard_df['Class']

# Initialize classifiers
classifiers = {
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "XGBClassifier": XGBClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42)
}

# Initialize the StratifiedKFold object
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # You can adjust the number of splits

# Dictionary to store accuracy scores and thresholds
results = {}

# Iterate over each classifier
for clf_name, clf in classifiers.items():
  # Lists to store accuracy scores and thresholds
  accuracy_scores = []
  thresholds = []
  # Iterate over each fold
  for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Fit the classifier on the training data
    clf.fit(X_train, y_train)
    # Predict probabilities on the test data
    y_proba = clf.predict_proba(X_test)[:, 1]
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    # Find the threshold that maximizes the F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = _[np.argmax(f1_scores)]
    # Predict using the optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    # Calculate accuracy score for this fold
    accuracy = accuracy_score(y_test, y_pred)
    # Append accuracy score and threshold to the lists
    accuracy_scores.append(accuracy)
    thresholds.append(optimal_threshold)
  # Calculate mean accuracy score and mean threshold
  mean_accuracy = np.mean(accuracy_scores)
  mean_threshold = np.mean(thresholds)
  # Store results in the dictionary
  results[clf_name] = {"Mean Accuracy Score": mean_accuracy, "Mean Threshold": mean_threshold}

st.title("Classifier Evaluation")
st.write("RESULTS")
for clf_name, result in results.items():
    st.write(clf_name)
    st.write("Accuracy Score:", result["Mean Accuracy Score"])
    st.write("Threshold:", result["Mean Threshold"])
    st.write()
