# Credit Score Prediction with Random Forest & Streamlit

An end-to-end machine learning project that predicts a customer's **credit score category — Poor, Standard, or Good** — from financial and credit-related information.

The project covers data preprocessing, exploratory analysis, model comparison, feature importance, modularization, and deployment through an interactive **Streamlit** application.

---

## 📌 Project Overview

The dataset contains **25,000 customer records and 29 original columns** covering personal information, income, banking activity, loans, payment history, and credit characteristics.

The target, `Credit_Score`, contains three classes:

* **Poor** — 29%
* **Standard** — 53%
* **Good** — 18%

The workflow consists of:

**Data Cleaning → EDA → Model Comparison → Feature Analysis → Modularization → Streamlit Deployment**

---

## 🧹 Data Preprocessing

Key preprocessing steps included:

* Cleaning inconsistent missing-value markers
* Removing identifiers such as `ID`, `Customer_ID`, `Name`, and `SSN`
* Cleaning numeric and categorical values
* Converting `Credit_History_Age` to months
* Converting loan information into a `Loan_Count`
* Handling unrealistic values
* Median/mode imputation
* One-hot encoding categorical features
* Standard scaling of numerical features
* Building preprocessing with scikit-learn pipelines

---

## 🤖 Model Comparison

Four classification models were evaluated using an **80/20 stratified train-test split**.

| Model               |   Accuracy | Macro F1 |
| ------------------- | ---------: | -------: |
| Logistic Regression |     0.6534 |     0.62 |
| Decision Tree       |     0.6468 |     0.62 |
| **Random Forest**   | **0.7498** | **0.73** |
| XGBoost             |     0.7278 |     0.71 |

**Random Forest achieved the best overall performance** and was selected as the primary deployment model.

### Random Forest Performance

| Class    | Precision | Recall |   F1 |
| -------- | --------: | -----: | ---: |
| Poor     |      0.77 |   0.74 | 0.75 |
| Standard |      0.77 |   0.78 | 0.78 |
| Good     |      0.66 |   0.67 | 0.66 |

Overall test accuracy: **74.98%**

---

## 🔎 Feature Importance

The most important features included:

* **Outstanding Debt**
* **Interest Rate**
* **Credit Mix**
* **Delay from Due Date**
* **Changed Credit Limit**
* **Occupation**
* **Credit History Age**

These features provided the strongest predictive signal for the Random Forest model.

---

## 🚀 Streamlit Deployment

The trained model was integrated into a Streamlit application where users can enter customer information and receive a predicted credit score.

Although the original dataset contains 29 columns, the application uses **18 practical user-input features**, covering income, banking activity, credit characteristics, debt, payment behavior, and loan information.

The application returns:

**🔴 Poor · 🟡 Standard · 🟢 Good**

🔗 **Live Application:** https://modeldeploymentuas-cgt1.streamlit.app/

### Architecture

```text
User Input
    ↓
Streamlit App
    ↓
CreditScorePredictor
    ↓
Preprocessing Pipeline
    ↓
Random Forest
    ↓
Prediction
```

---

## 🛠️ Tech Stack

**Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · Matplotlib · Seaborn**

Key techniques:

`Random Forest` · `XGBoost` · `Pipeline` · `ColumnTransformer` · `OneHotEncoder` · `StandardScaler`

---

## 📁 Project Structure

```text
Credit Score Prediction with Random Forest & Streamlit/
├── app/
│   └── streamlit_app.py
├── data/
│   └── processed/
├── models/
├── src/
│   ├── inference.py
│   └── preprocessing.py
├── notebooks/
├── train.py
├── train_streamlit.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

* The dataset contains some suspicious or unrealistic financial values.
* The **Good** class is underrepresented and has the lowest F1-score.
* The deployed application uses 18 practical inputs rather than every original dataset column.
* Predictions are machine learning estimates and **should not be treated as real-world credit or financial decisions**.

---

## 🔮 Future Improvements

* Move all preprocessing strictly inside the training pipeline.
* Address class imbalance using class weights or resampling.
* Perform more extensive hyperparameter tuning and cross-validation.
* Add prediction probabilities/confidence scores to the Streamlit interface.
* Further investigate anomalous financial values.
