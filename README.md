# 💳 Real-Time E-Commerce Fraud Detection

A machine learning project that detects potentially fraudulent e-commerce transactions using transaction, customer, payment, device, and merchant-related information.

The project demonstrates a complete machine learning workflow, including data exploration, data cleaning, preprocessing, model training, evaluation, model persistence, and a Streamlit-based prediction application.

## 📌 About the Project

Online transactions can contain fraudulent activities such as unauthorized purchases, suspicious payment behavior, unusual transaction amounts, and abnormal customer activity.

This project aims to build a machine learning system that analyzes transaction information and predicts whether a transaction is:

- `0` → Legitimate Transaction
- `1` → Fraudulent Transaction

The project is designed as an end-to-end machine learning application rather than only a model-training experiment.

## 🎯 Project Objective

The main objectives of this project are:

- Analyze transaction data using Exploratory Data Analysis
- Identify missing values and duplicate records
- Clean and preprocess the dataset
- Analyze the fraud/legitimate class distribution
- Prepare numerical and categorical features for machine learning
- Train a fraud classification model
- Evaluate model performance using multiple metrics
- Save the trained model for later use
- Build a simple application for real-time transaction prediction

## 🗂️ Dataset

The dataset used in this project is **synthetic data created for educational and machine learning development purposes**.

It represents an e-commerce transaction environment and contains information related to:

- Transaction amount
- Customer information
- Account information
- Previous transaction activity
- Failed transaction activity
- Payment method
- Device type
- Merchant category
- Fraud status

### Target Variable

```text
is_fraud
