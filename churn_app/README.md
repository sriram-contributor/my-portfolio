![Project Status](https://img.shields.io/badge/status-completed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

# 📊 Customer Churn Prediction using Machine Learning

This project aims to predict customer churn using machine learning models. Churn prediction helps businesses retain valuable customers by proactively identifying those likely to leave. 

This project is an end-to-end machine learning solution to predict customer churn for a telecom company, built using Python and deployed as a live web application with Streamlit.

🔗 **Live App**: [Click to try it out!](https://my-portfolio-qztukjwt6wvent5azdenar.streamlit.app/)  
📁 **GitHub Repo**: [https://github.com/sriram-contributor/my-portfolio](https://github.com/sriram-contributor/my-portfolio)

---

## 🔍 Problem Statement

Given historical data from a telecom company, the goal is to build a predictive model to classify whether a customer will churn.

## 📁 Dataset

- Source: [IBM Sample Telco Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Rows: 7,043
- Target: `Churn` (Yes/No)

## 🧪 Workflow

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model training (Logistic Regression, Random Forest, XGBoost)
5. Evaluation using Accuracy, F1-score, ROC-AUC
6. Interpretability using SHAP and feature importance

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.804    | 0.648     | 0.575  | 0.609    | 0.836   |
| Random Forest       | 0.788    | 0.623     | 0.513  | 0.563    | 0.816   |
| XGBoost             | 0.767    | 0.568     | 0.516  | 0.541    | 0.814   |

> Logistic Regression outperformed the more complex models, indicating the problem may be linearly separable and well-suited for simpler models in this case.

## 🧠 Key Insights

- Customers on month-to-month contracts churn more frequently.
- Short tenure and use of electronic check payments are strong indicators of churn.
- SHAP values and feature importance were used to interpret model predictions and uncover key risk factors.

## 🛠 Tools & Libraries

- Python, Pandas, Scikit-learn, XGBoost
- Matplotlib, Seaborn, SHAP
- Jupyter Notebook

## 🗂 Folder Structure

churn_app/
├── app.py # Streamlit app \
├── model.pkl # Trained ML model \
├── scaler.pkl # Scaler used on input features \
├── feature_columns.pkl # List of all features used in training \
├── requirements.txt # Required Python packages


---

## 💡 How to Run This App Locally

```bash
git clone https://github.com/sriram-contributor/my-portfolio.git
cd churn_app
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Future Improvements

- Deploy the model with a web interface (e.g., Streamlit)
- Integrate with a CRM dashboard to track real-time churn risk
- Explore deep learning models or ensemble stacking approaches

## 🔗 Author

Sriram L  
[LinkedIn](https://linkedin.com/in/sriram-lourdu/)  
[GitHub](https://github.com/sriram-contributor)
