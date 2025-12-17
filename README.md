# Machine Learning Classical Models vs Deep Learning Neural Network
## Loan-Default-Prediction


### **Team Members:**

- **Muhammad Muneeb** 

- **Hassan Shakeel**
---


### **Abstract**


Loan default prediction is a critical task for financial institutions to minimize credit risk and improve decision-making. This project focuses on predicting whether a borrower will default on a loan using financial and demographic attributes. Classical machine learning models and deep learning approaches are implemented and compared using multiple evaluation metrics. Experimental results show that deep learning models outperform traditional approaches, offering improved recall and reduced financial risk.


### **Introduction**

Loan default prediction helps banks and lending institutions assess the risk associated with granting loans. Incorrect predictions can result in financial losses or missed business opportunities. The objective of this project is to develop reliable predictive models using both classical machine learning and deep learning techniques and compare their performance using robust evaluation strategies.


### **Dataset Description**

- **Dataset Name:** Loan_default.csv

- **Source:** Public financial loan dataset

- **Records:** Loan related information on more than 250,000+ individuals

- **Target Variable:** Loan Default (0 = No Default, 1 = Default)
  
- **Features:**
Demographic attributes (age, income, employment type),


Financial attributes (loan amount, interest rate, credit score)

#### **Preprocessing Steps**

- Handling missing values

- Encoding categorical variables

- Feature scaling using StandardScaler

- Train-test split with fixed random seed


### **Methodology**
Classical Machine Learning Models:

- Logistic Regression

- Decision Tree Classifier


#### **Techniques Used:**

- Feature engineering

- Hyperparameter tuning


### **Deep Learning Model**

- Fully connected neural network using TensorFlow

- Hidden layers with ReLU activation

- Regularization using Dropout

- Learning rate Scheduling


### **Results & Analysis**

The performance of the implemented models was evaluated using multiple metrics, including accuracy, recall, F1-score, and AUC. Due to the imbalanced nature of the dataset, special emphasis was placed on the default class (class 1), as it represents high-risk borrowers. The following table summarizes the comparative performance of the Logistic Regression, Decision Tree, and Neural Network models.

#### **Model Performance Comparison**

| Model | Accuracy | Recall (Default) | F1-Score (Default) | AUC |
|------|---------|------------------|--------------------|-----|
| Logistic Regression | 0.8853 | 0.03 | 0.0645 | — |
| Decision Tree | 0.8849 | — | 0.0420 | — |
| Neural Network | 0.8846 | 0.0693 | 0.1224 | 0.7542 |



Although all models achieved similar accuracy values (88%), accuracy alone is misleading due to the strong class imbalance in the dataset. Logistic Regression and Decision Tree classifiers achieved high accuracy by predominantly predicting the majority class (non-default), resulting in extremely low recall and F1-scores for the default class. In contrast, the neural network model demonstrated superior performance in identifying defaulters, achieving the highest F1-score (0.1224) and recall (0.0693) for the default class. This highlights the effectiveness of deep learning in capturing complex non-linear relationships in imbalanced financial datasets.

Low recall and F1-score are expected in imbalanced credit-risk datasets when the model prioritizes overall accuracy or “No Default” predictions.


### **Business Impact Analysis**

From a business perspective, false negatives are costly as they result in approving high-risk borrowers. Although Logistic Regression and Decision Tree models achieved high accuracy (88%), their very low recall for defaulters (as low as 0.03) shows poor risk detection. The neural network achieved the highest recall (0.0693) and F1-score (0.1224) for the default class, making it more effective for identifying risky borrowers despite a slight drop in overall accuracy.


### **Conclusion & Future Work**

This project demonstrates that deep learning models provide superior performance in loan default prediction compared to classical machine learning approaches. Future work may include incorporating additional financial features, using ensemble learning, and deploying the model as a real-time decision support system.

### **References**

- Scikit-learn Documentation

- TensorFlow Documentation

- Kaggle Financial Datasets
