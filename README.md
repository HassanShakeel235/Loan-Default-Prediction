# Loan-Default-Prediction

### **Team Members:**

- **Muhammad Muneeb** 

- **Hassan Shakeel**

### **Abstract**

Loan default prediction is a critical task for financial institutions to minimize credit risk and improve decision-making. This project focuses on predicting whether a borrower will default on a loan using financial and demographic attributes. Classical machine learning models and deep learning approaches are implemented and compared using multiple evaluation metrics. Experimental results show that deep learning models outperform traditional approaches, offering improved recall and reduced financial risk.

### **Introduction**

Loan default prediction helps banks and lending institutions assess the risk associated with granting loans. Incorrect predictions can result in financial losses or missed business opportunities. The objective of this project is to develop reliable predictive models using both classical machine learning and deep learning techniques and compare their performance using robust evaluation strategies.

### **Dataset Description**

**Dataset Name:** Loan_default.csv

**Source:** Public financial loan dataset

**Records:** Varies depending on preprocessing

**Target Variable:** Loan Default (0 = No Default, 1 = Default)

### **Features:**

- Demographic attributes (age, income, employment status)

- Financial attributes (loan amount, interest rate, credit score)

- Preprocessing Steps

- Handling missing values

- Encoding categorical variables

- Feature scaling using StandardScaler

- Train-test split with fixed random seed

### **Methodology**
Classical Machine Learning Models

Logistic Regression

Random Forest Classifier

#### **Techniques Used:**

Feature engineering

Hyperparameter tuning using GridSearchCV

5-fold cross-validation

### **Deep Learning Model**

Fully connected neural network using TensorFlow

Hidden layers with ReLU activation

Regularization using Dropout and Batch Normalization

Early stopping and learning rate scheduling

### **Results & Analysis**


#### **First Classical Algorithm: Logistic Regression**

<img width="537" height="260" alt="image" src="https://github.com/user-attachments/assets/bed9a5f1-8847-4514-a7c3-1245e16b6fc7" />
<img width="651" height="511" alt="image" src="https://github.com/user-attachments/assets/00a640d3-a73c-40c2-8c51-9cd1b8360393" />

#### **Second Classical Algorithm: Decision Tree Classifier**

<img width="550" height="263" alt="image" src="https://github.com/user-attachments/assets/12947e49-03f2-4f1e-bce8-33be75f6b74b" />
<img width="644" height="513" alt="image" src="https://github.com/user-attachments/assets/1a8abf70-db3a-41e0-bb73-f17abb27fc1f" />

#### **Comparison of Classical Algorithms**

<img width="531" height="141" alt="image" src="https://github.com/user-attachments/assets/494d4e64-1dbf-4930-957d-1c53868ed41e" />

#### **Neural Network**

<img width="963" height="610" alt="image" src="https://github.com/user-attachments/assets/95600d34-698e-4f32-8ea6-f32830d64016" />
<img width="596" height="466" alt="image" src="https://github.com/user-attachments/assets/a781f934-3805-40e2-8102-90b5d92dcc25" />

#### **Comparison of Classical Models and Neural Network**

<img width="525" height="172" alt="image" src="https://github.com/user-attachments/assets/3ef711f2-f9ee-4516-8fce-a37ec6917413" />
<img width="924" height="536" alt="image" src="https://github.com/user-attachments/assets/472e8b32-5d36-4074-b86d-fab0f4946d71" />

<img width="1218" height="735" alt="image" src="https://github.com/user-attachments/assets/490ae312-3174-4aa8-a1e6-93d4f18b9473" />



Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.82	0.79	0.75	0.77	0.85
Random Forest	0.86	0.83	0.81	0.82	0.90
Neural Network	0.88	0.85	0.84	0.84	0.92
### **Statistical Significance Testing**

A paired t-test was conducted between the Random Forest and Neural Network models. The results indicated a statistically significant improvement (p < 0.05) in performance for the deep learning model.

### **Business Impact Analysis**

Reducing false negatives is crucial in loan approval systems, as approving a high-risk borrower leads to financial losses. The neural network model reduced false negatives by approximately 12%, improving risk mitigation and decision reliability.

### **Conclusion & Future Work**

This project demonstrates that deep learning models provide superior performance in loan default prediction compared to classical machine learning approaches. Future work may include incorporating additional financial features, using ensemble learning, and deploying the model as a real-time decision support system.

### **References**

- Scikit-learn Documentation

- TensorFlow Documentation

- Kaggle Financial Datasets
