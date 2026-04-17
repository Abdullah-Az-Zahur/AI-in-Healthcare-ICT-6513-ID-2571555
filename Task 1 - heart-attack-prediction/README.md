# ❤️ Heart Attack Prediction Machine Learning

## 📌 Project Overview

This project is developed as part of the course **ICT 6513 – AI in Healthcare**. The main objective is to predict the likelihood of heart disease (heart attack risk) using machine learning techniques based on clinical health data.

The model analyzes various health-related features such as age, cholesterol level, blood pressure, and maximum heart rate to determine whether a patient is at risk.

---

## 🎯 Objectives

* Predict heart disease presence (Yes/No)
* Analyze important health factors affecting heart disease
* Visualize model performance using graphs

---

## 📊 Dataset

The dataset contains the following features:

* age – Age of the patient
* sex – Gender
* cp – Chest pain type
* trestbps – Resting blood pressure
* chol – Cholesterol level
* fbs – Fasting blood sugar
* restecg – Resting ECG results
* thalach – Maximum heart rate achieved
* exang – Exercise-induced angina
* oldpeak – ST depression
* slope – Slope of ST segment
* ca – Number of major vessels
* thal – Thalassemia
* target – Heart disease (0 = No, 1 = Yes)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 🤖 Machine Learning Model

* Algorithm: **Logistic Regression**
* Type: Classification

The model is trained to classify whether a patient has heart disease or not.

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

* Accuracy Score
* Confusion Matrix
* ROC Curve (AUC)
* Feature Importance Visualization

---

## 📊 Dashboard Visualization

A dashboard-style visualization is implemented using Matplotlib, including:

* Confusion Matrix
* Accuracy Bar Chart
* Feature Importance Graph
* ROC Curve

---

## 🚀 How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/Abdullah-Az-Zahur/heart-attack-prediction
cd heart-attack-prediction
```

### 2. Create Virtual Environment

```
python -m venv venv
```

### 3. Activate Environment

**Windows:**

```
venv\Scripts\activate
```

**Mac/Linux:**

```
source venv/bin/activate
```

### 4. Install Dependencies

```
pip install pandas numpy matplotlib scikit-learn
```

### 5. Run the Project

```
python main.py
```

---

## 📌 Results

The model successfully predicts heart disease with good accuracy. The dashboard provides a clear visualization of model performance and feature impact.

---

## 🧠 Real-World Application

This type of machine learning model can assist healthcare professionals in:

* Early detection of heart disease
* Risk assessment of patients
* Supporting clinical decision-making

---

## 🙏 Acknowledgment

This project was completed as part of academic coursework under **ICT 6513 – AI in Healthcare**.

---

## 📧 Author

**Md. Abdullah Az-Zahur**

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
