# Credit Card Fraud Detection

## Overview

This project builds a machine learning model to detect fraudulent credit card transactions. The model is trained using a credit card transaction dataset and compares the performance of three machine learning algorithms:

* Logistic Regression
* Decision Tree
* Random Forest

The objective is to classify each transaction as either **Fraudulent** or **Legitimate**.

---

## Dataset

**Source:** Kaggle - Credit Card Fraud Detection Dataset

https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset contains:

* `fraudTrain.csv`
* `fraudTest.csv`

---

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluate.py
│
├── Credit_Card_Fraud_Detection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

* Data loading using Pandas
* Data preprocessing
* Feature engineering
* Categorical feature encoding
* Logistic Regression model
* Decision Tree model
* Random Forest model
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
  * Classification Report

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Feature Engineering

The following preprocessing steps were performed:

* Removed unnecessary columns
* Converted transaction date into:

  * Transaction Hour
  * Transaction Day
  * Transaction Month
* Converted Date of Birth into Customer Age
* Encoded categorical features using OrdinalEncoder

---

## Models Used

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 99.61%   |
| Decision Tree       | 99.68%   |
| Random Forest       | 99.86%   |

Random Forest achieved the best overall performance for fraud detection.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Credit_Card_Fraud_Detection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

Run the notebook cells sequentially.

---

## Author
Saad Siddiqui

Developed as a Machine Learning project for Credit Card Fraud Detection.
