import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Count the number of fraudulent and non-fraudulent transactions
class_counts = data['Class'].value_counts()

# Create a bar chart
fig_bal_bar, ax = plt.subplots()
ax.bar(class_counts.index.map({0: 'Non-Fraudulent', 1: 'Fraudulent'}), class_counts.values, color=['skyblue', 'skyblue'])
ax.set_title('Number of Non-Fraudulent vs. Fraudulent Transactions')
ax.set_xlabel('Class')
ax.set_ylabel('Count')



# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Make predictions
y_pred = model.predict(X_test)

# Create confusion matrix
cm1 = confusion_matrix(y_test, y_pred)

