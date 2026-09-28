# Fraud Detection Dataset

This directory contains the synthetic transaction dataset used by the
**Real-Time E-Commerce Fraud Detection System**.

The dataset is generated programmatically using the dataset generator
located in the project and is designed for experimenting with fraud
detection, data preprocessing, exploratory data analysis, and machine
learning models.

> **Important:** This is a synthetic dataset created for educational,
> experimentation, and machine-learning development purposes. It does not
> contain real customer, merchant, or financial transaction data.

## Dataset Overview

The generator is configured to create:

- **500,000 requested transaction records**
- Approximately **50% fraudulent transactions**
- Approximately **50% legitimate transactions**
- Multiple customer, transaction, merchant, device, behavioral, and risk features
- Missing values
- Outliers
- Invalid numerical values
- Zero values
- Categorical inconsistencies
- Duplicate transaction records
- Duplicate transaction IDs
- Randomized transaction timestamps

The dataset is intentionally imperfect so that the preprocessing pipeline
can simulate some of the data-quality problems found in real-world
transaction systems.

## Class Distribution

The current generator uses:

```python
TARGET_RATE = 0.50
