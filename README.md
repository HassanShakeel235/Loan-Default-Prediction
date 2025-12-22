# Machine Learning Classical Models vs Deep Learning Neural Network
## Loan-Default-Prediction


### **Team Members:**

- **Muhammad Muneeb** 

- **Hassan Shakeel**
---


### **Abstract**


Loan default prediction is a critical task for financial institutions to minimize credit risk and improve decision-making. This project focuses on predicting whether a borrower will default on a loan using financial and demographic attributes. Classical machine learning models and deep learning approaches are implemented and compared using multiple evaluation metrics. Experimental results show that one of the two classical model outperforms deep learning model , offering improved recall and reduced financial risk.


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

Logistic Regression

- Automated Pipeline: Used a ColumnTransformer to simultaneously handle mean-imputation for missing values, Standard Scaling for numerical data, and One-Hot Encoding for categories.
- Stratified Sampling: Performed an 80/20 data split using stratify=y to ensure the model trained on a balanced representation of both "Default" and "No Default" cases.
- Multidimensional Evaluation: Beyond simple accuracy, the methodology focuses on the F1-Score for the "Default" class to ensure a balance between precision and recall for high-risk predictions.
  

Decision Tree Classifier
- Non-Linear Classification: Implemented a Decision Tree Classifier, which captures complex, non-linear relationships between features (like Credit Score and Loan Amount) that Logistic Regression might miss.
- Visual Diagnostics: Utilized Seaborn-based Heatmaps to analyze the confusion matrix, specifically tracking how many "Defaults" were correctly identified versus misclassified.
- Performance Benchmarking: Stored F1-Scores for both classes into a global results dictionary to allow for a direct head-to-head comparison with the Logistic Regression model.


#### **Techniques Used:**
- Early Stopping
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

| Model | Accuracy | Recall (Default) | F1-Score (Default) | 
|------|---------|------------------|--------------------|
| Logistic Regression | 0.8005 | 0.4986 | 0.3672 |
| Decision Tree | 0.7584 | 0.5013 | 0.3252 |
| Neural Network | 0.8349 | 0.4001 | 0.3503 |


Among the three models, the Neural Network achieves the highest accuracy (83.49%), indicating better overall classification performance. Logistic Regression also performs well with an accuracy of 80.05%, while the Decision Tree shows the lowest accuracy at 75.84%. However, when focusing on the F1-score for the default class, Logistic Regression performs best (0.3672), reflecting a better balance between precision and recall. The Neural Network, despite its high accuracy, has a lower F1-score (0.3503), suggesting weaker default-class detection. Overall, Logistic Regression provides the most balanced performance for identifying the default class.

Low recall and F1-score are expected in imbalanced credit-risk datasets when the model prioritizes overall accuracy or “No Default” predictions.


### **Business Impact Analysis**

From a business perspective, false negatives are costly as they lead to approving high-risk borrowers. Although the Neural Network achieved the highest overall accuracy (83.49%), accuracy alone is not sufficient for risk assessment. When focusing on the default class, Logistic Regression shows the best balance, achieving the highest F1-score (0.3672) with a competitive recall (0.4986), indicating more reliable identification of defaulters. The Decision Tree has similar recall (0.5013) but a noticeably lower F1-score (0.3252), suggesting less consistent predictions. Overall, Logistic Regression emerges as the most effective model for detecting risky borrowers.


### **Conclusion & Future Work**

This project demonstrates that while deep learning models can achieve higher overall accuracy in loan default prediction, superior accuracy does not necessarily translate to better detection of defaulters. Despite the Neural Network attaining the highest accuracy, its recall and F1-score for the default class remain lower than those of Logistic Regression, primarily due to the severe class imbalance in the dataset. Classical machine learning models, particularly Logistic Regression, show a more balanced performance for identifying risky borrowers. Future work may focus on addressing class imbalance through resampling techniques, incorporating additional financial features, exploring ensemble learning methods, and deploying the system as a real-time decision support tool.

### **References**

- Scikit-learn Documentation

- TensorFlow Documentation

- Kaggle Financial Datasets
