# Loan Approval Prediction

A supervised machine learning project that predicts whether a loan application will be approved, based on applicant financial details. Three classification models are trained and compared using precision, recall, F1-score, accuracy and confusion matrix.

## Problem Statement
Banks receive many loan applications and must decide quickly which ones to approve. Approving a loan that later defaults is costly, so this project builds a classification model that predicts loan approval (`Loan_Approved`) while paying special attention to **precision**.

## Dataset
- **File:** `loan_approval_data.csv`
- **Size:** about 1000 rows (200 used for testing, 800 for training)
- **Target variable:** `Loan_Approved`
- **Key features:** `Credit_Score`, `DTI_Ratio` (debt-to-income ratio), `Applicant_Income`, [add other columns]

## Project Workflow
1. **Data Cleaning and EDA:** [add what you did]
2. **Feature Engineering:**
   - Created squared features: `DTI_Ratio_sq` and `Credit_Score_sq` to capture non-linear effects
   - Applied a log transformation (`log1p`) to `Applicant_Income` to reduce skewness
   - Dropped the original `Credit_Score` and `DTI_Ratio` columns after creating the squared versions
3. **Train-Test Split:** 80% training, 20% testing (`random_state=42`)
4. **Feature Scaling:** `StandardScaler` (fit on training data only, to avoid data leakage)
5. **Model Building:** Logistic Regression, K-Nearest Neighbors (k=11), Gaussian Naive Bayes
6. **Evaluation:** Precision, Recall, F1-score, Accuracy and Confusion Matrix

## Results

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Logistic Regression | 0.783 | 0.770 | 0.777 | 86.5% |
| KNN (k=11) | 0.675 | 0.443 | 0.535 | 76.5% |
| Naive Bayes | **0.804** | 0.738 | 0.769 | 86.5% |

**Confusion matrices** (rows = actual, columns = predicted):

| Model | True Negative | False Positive | False Negative | True Positive |
|-------|---------------|----------------|----------------|---------------|
| Logistic Regression | 126 | 13 | 14 | 47 |
| KNN | 126 | 13 | 34 | 27 |
| Naive Bayes | 128 | 11 | 16 | 45 |

**Best model: Naive Bayes**, selected on the basis of precision (0.804). It makes the fewest false approvals (11), which matters most for a bank, since wrongly approving a risky loan is costly. Logistic Regression is a close second, with slightly better recall and F1-score.

## Key Insights
- Naive Bayes and Logistic Regression both reached 86.5% accuracy and clearly outperformed KNN.
- KNN struggled to identify approved loans (recall of only 0.44), missing 34 of 61 actual approvals.
- Precision was used as the main metric because false approvals are more costly than false rejections.
- The dataset is imbalanced (about 70% rejected, 30% approved in the test set), so accuracy alone can be misleading.

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run
1. Clone the repository
   ```
   git clone https://github.com/rutujakhade23/AI-ML-Practice.git
   ```
2. Go to the project folder
   ```
   cd AI-ML-Practice/Supervised-ML/MinorProject
   ```
3. Install the required libraries
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
4. Open the notebook
   ```
   jupyter notebook credit_wise.ipynb
   ```

## Future Improvements
- Hyperparameter tuning with GridSearchCV
- Handle class imbalance using SMOTE or class weights
- Try ensemble models such as Random Forest and Gradient Boosting
- Use cross-validation for more reliable results
- Deploy as a web app using Streamlit

## Author
**Rutuja Khade**
GitHub: [rutujakhade23](https://github.com/rutujakhade23)
