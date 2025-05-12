# StyleSense Review Recommendation Model

Welcome to the **StyleSense** machine learning project! This project aims to predict whether a customer would recommend a clothing item based on their review, age, product category, and other metadata.

---

## 📌 Objective

StyleSense, a growing online women's fashion retailer, is facing a challenge: many customer reviews are missing the "Recommended" label. Our goal is to use reviews with complete data to train a machine learning model that can predict this missing information. This will help:
- Improve product recommendations
- Understand customer sentiment
- Identify top-performing items

---

## 📁 Dataset Overview

The dataset `reviews.csv` contains 18,442 customer reviews with the following fields:

| Column Name                | Description                                   |
|----------------------------|-----------------------------------------------|
| `Clothing ID`              | Unique ID for each product                    |
| `Age`                      | Age of the customer                           |
| `Title`                    | Short review title                            |
| `Review Text`              | Full customer review                          |
| `Positive Feedback Count`  | Count of helpful votes for a review           |
| `Division Name`            | Broad category (e.g., General, Petite)        |
| `Department Name`          | Specific department (e.g., Tops, Dresses)     |
| `Class Name`               | Product class (e.g., Knits, Blouses)          |
| `Recommended IND`          | **Target** — 1 = Recommended, 0 = Not         |

---

## 🔍 EDA (Exploratory Data Analysis)

We performed EDA to understand:
- Distribution of age and feedback
- Frequency of each product category
- Text length in reviews
- Class imbalance in recommendation labels

---

## 🧪 Model Pipeline

We built a machine learning pipeline using `scikit-learn`:

- **Numerical Features**: Scaled with `StandardScaler`
- **Categorical Features**: Encoded using `OneHotEncoder`
- **Text Reviews**: Transformed with `TfidfVectorizer`
- **Feature Engineering**: `SpacyLemmatizer`, `CharacterCount()`
- **Model**: `RandomForestClassifier` 
- **Model Evaluation Metric**: `Accuracy` 
- **Fine-Tuning**: `RandomizedGridSearch` 

---
