Problem statement
The majority of them now use credit cards to purchase items that they desperately need but are currently unable to afford. 
Credit cards are used to satisfy needs, but there is also an increase in credit card fraud, thus it is necessary to create a model that fits well and makespredictions with more accuracy.

Objectives
• The main objective of the research is to find a fraudulent transactions in credit card
transactions.
• When supervised learning and deep learning were compared, the deep learning
algorithm performed better in terms of accuracy

About Data 
The dataset used by the suggested system was downloaded from 
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. The transactions that customers 
made in a European bank throughout the 2017–18 year comprisethe dataset. It has 31 
columns total, of which 30 are features and 1 is the target class that determines if the 
transaction is fraudulent or not.

Conclusion
In conclusion, our study demonstrates the effectiveness of machine learning algorithms in
detecting credit card fraud. We achieved promising results some various machine learning
models in which The XGBOOST model with Random Oversampling with Stratified KFold CV provided us with the best accuracy and ROC on oversampled data out of all the
models we built in the oversample situations.
Following that, we adjusted the hyperparameters and obtained the metrics But out of all
the models we developed, we discovered that the best outcome was obtained using
Logistic Regression with L2Regularization for Stratified K-Fold cross validation (without
any oversampling or under sampling).
The analysis highlighted the importance of transaction amount, time, and frequency in
fraud detection. While our models performed well, future research could explore deep
learning and anomaly detection methods for further improvements.
Overall, our study contributes to combating credit card fraud and shows the potential of
machine learning in enhancing fraud detection systems for financial institutions
