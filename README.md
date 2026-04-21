# 🍜 Street Food Vendor Survival Predictor

An end-to-end machine learning application that predicts the survival probability of urban street food vendors using real-world business and environmental factors.

---

## 📌 Overview

This project builds a complete ML pipeline—from data preprocessing and feature engineering to model deployment—wrapped in an interactive web application.

The goal is to provide **actionable insights** for street food vendors and stakeholders to improve business success rates.

---


---
```markdown
## 🚀 Live Demo

*Deployment in progress. You can run the app locally:*

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
---

## 🧠 Problem Statement

Street food vendors operate in highly competitive and uncertain environments.
This project aims to answer:

> **Can we predict whether a vendor will succeed based on measurable factors?**

---

## ⚙️ Features

* ✅ End-to-end ML pipeline (data → model → deployment)
* ✅ Advanced feature engineering
* ✅ Hyperparameter tuning using Optuna
* ✅ Model interpretation using SHAP
* ✅ Interactive Streamlit web app
* ✅ Real-time predictions

---

## 🧪 Model Details

* **Algorithm:** XGBoost Classifier
* **Evaluation Metric:** ROC-AUC
* **Optimization:** Optuna hyperparameter tuning
* **Interpretability:** SHAP analysis

---

## 📊 Input Features

The model considers multiple business and environmental factors, including:

* Vendor demographics (age, experience)
* Daily revenue & customer flow
* Competition density
* Operational hours
* Location type
* Health inspection scores
* Online presence

---

## 📈 Output

* 🔹 Survival Probability (0–1)
* 🔹 Business Success Prediction

---

## 🏗️ Project Structure

```
AIML/
├── deployment/
│   ├── app.py
│   ├── feature_engineering.py
│   ├── test_model.py
│   ├── requirements.txt
│   └── model/
│       ├── xgb_model.pkl
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       ├── feature_names.json
│       └── threshold.json
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Key Insights

* Vendors with higher **customer flow per hour** show better survival rates
* **Competition density** significantly impacts performance
* **Operational efficiency metrics** (revenue per helper, per hour) are strong predictors

---

## 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* Optuna
* SHAP
* Streamlit

---

## 🎯 Future Improvements

* Deploy as a REST API
* Add real-time data integration
* Improve UI/UX design
* Incorporate geospatial analysis

---

## 🤝 Contributing

Feel free to fork the repository and submit pull requests.

---

## 📬 Contact

*(Add your LinkedIn / Email here)*

---

## ⭐ If you found this useful

Give the repo a ⭐ to support the project!
# street-food-vendor-survival-predictor
