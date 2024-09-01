import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import credit as cd
import evalution as ev

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Count the number of fraudulent and non-fraudulent transactions
class_counts = data['Class'].value_counts()

# Created a bar chart for in-balanced data
fig_bar, ax = plt.subplots()
ax.bar(class_counts.index.map({0: 'Non-Fraudulent', 1: 'Fraudulent'}), class_counts.values, color=['skyblue', 'skyblue'])
ax.set_title('Number of Non-Fraudulent vs. Fraudulent Transactions')
ax.set_xlabel('Class')
ax.set_ylabel('Count')



# Created a pie chart for in-balanced data
fig_pie, ax = plt.subplots()
ax.pie(class_counts.values, labels=class_counts.index.map({0: 'Non-Fraudulent', 1: 'Fraudulent'}), colors=['skyblue', 'lightcoral'], autopct='%1.1f%%', startangle=140)
ax.set_title('Proportion of Non-Fraudulent vs. Fraudulent Transactions')



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
cm = confusion_matrix(y_test, y_pred)

#---------------------------------------------------------------------------------------

# create Streamlit app
st.title("Credit Card Fraud Detection Using Machine Learning")
st.write("Enter the following features to check if the transaction is Non-fraudulent or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input the Data')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")
#acc = st.button("Evalution")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Non-Fraudulent transaction")
    else:
        st.write("Fraudulent transaction")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit app
st.title("Data Visualization")

# Add a radio button to choose visualization For Unbalanced Data
selected_option = st.radio("Select Visualization For Unbalanced Data: ", ["Confusion Matrix", "Bar Chart", "Pie Chart"])
st.set_option('deprecation.showPyplotGlobalUse', False)
if selected_option == "Confusion Matrix":
    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Pass plt object to st.pyplot()
    st.pyplot()

elif selected_option == "Bar Chart":
    # Display the bar chart using Streamlit
    st.pyplot(fig_bar)

elif selected_option == "Pie Chart":
    # Display the pie chart using Streamlit
    st.pyplot(fig_pie)

selected_option2 = st.radio("Select Visualization For Balanced Data: ", ["Confusion Matrix", "Bar Chart"])

if selected_option2 == "Confusion Matrix":
    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cd.cm1, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # Pass plt object to st.pyplot()
    st.pyplot()

elif selected_option2 == "Bar Chart":
    # Display the bar chart using Streamlit
    st.pyplot(cd.fig_bal_bar)



st.title("Classifier Evaluation")
# Add a radio button to choose classifier
selected_classifier3 = st.radio("Select Classifier", list(ev.conf_matrices.keys()))

# Show results and confusion matrix for the selected classifier
if st.button("Show Results"):
    st.title("Results for " + selected_classifier3)
    for clf_name, result in ev.results.items():
        if clf_name == selected_classifier3:
            st.write(clf_name)
            st.write("Accuracy Score:", result["Mean Accuracy Score"])
            st.write("Threshold:", result["Mean Threshold"])

    # Show confusion matrix for the selected classifier
    st.write("Confusion Matrix for", selected_classifier3)
    cm = ev.conf_matrices[selected_classifier3]
    # Calculate proportions for better visualization
    cm_proportions = cm / cm.sum(axis=1)[:, np.newaxis]
    # Create a heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cm_proportions, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, linecolor="black", ax=ax)
    st.pyplot(fig)
