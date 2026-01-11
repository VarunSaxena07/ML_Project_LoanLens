# LoanLens – Loan Approval Prediction System 

LoanLens is a machine learning project that predicts loan approval outcomes
based on applicant financial, personal, and credit-related information.
The goal is to analyze risk factors and build a reliable prediction system
using supervised learning techniques.

## Project Structure
- **EDA.ipynb** – Exploratory data analysis and initial insights
- **LoanLens_LogisticRegression.ipynb** – Logistic Regression model training and evaluation
- **LoanLens_KNN.ipynb** – K-Nearest Neighbors model training and evaluation
- **LoanLens_Naive.ipynb** – Naive Bayes model training and evaluation
- **LoanLens_Full.ipynb** – Complete end-to-end ML pipeline
- **loan_approval_data.csv** – Dataset used for training and evaluation

## Models Used
The following models were trained and evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

Model performance was compared using accuracy, precision, recall, F1-score,
and confusion matrix.

## Final Model Selection
Based on comparative evaluation across multiple metrics, **Naive Bayes**
demonstrated the most consistent performance and was selected as the final
model for deployment.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter Notebook

## Future Work
- Deploy the final model using Streamlit
- Improve feature engineering
- Add model interpretability and UI enhancements


## Key Learnings

During this project, I learned and applied several important machine learning
and data preprocessing concepts, including:

- Handling missing values using **SimpleImputer** with appropriate strategies
  for numerical and categorical features
- Encoding categorical variables using **LabelEncoder** and **OneHotEncoder**
  based on feature type
- Performing **correlation analysis** and visualizing relationships between
  numerical features using correlation heatmaps
- Understanding the impact of feature scaling using **StandardScaler**
- Comparing multiple machine learning models to select the most suitable one
  for deployment
