# Telco Customer Churn Prediction using XGBoost

## Project Overview

This project focuses on predicting customer churn for a telecommunications company using the Telco Customer Churn dataset. Churn prediction is a critical task for businesses to proactively retain customers and reduce revenue loss.

The machine learning model utilized here is **XGBoost (Extreme Gradient Boosting)**. XGBoost is an advanced implementation of gradient boosting designed for speed and performance. It builds an ensemble of decision trees sequentially, where each new tree attempts to correct errors from previous trees. Key features include:

- **High efficiency and scalability:** Optimized for fast training on large datasets.
- **Regularization techniques:** Helps prevent overfitting, improving model generalization.
- **Automatic handling of missing data:** Reduces the need for extensive preprocessing.
- **Support for weighted data:** Useful in handling class imbalance through parameters like `scale_pos_weight`.
- **Flexibility:** Supports various objective functions and evaluation metrics, making it adaptable to different tasks.

## Dataset

- Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Contains 7,043 customer records with demographic, account, and service-related features.
- Target variable: `Churn` (Yes/No), indicating whether a customer has discontinued service.

## Key Steps

1. **Exploratory Data Analysis (EDA)**  
   Initial exploration to understand data distribution, missing values, and feature types.

2. **Data Preprocessing**  
   - Separation of numeric and categorical features.  
   - Label encoding of categorical variables to convert them into numeric form.  
   - Combining processed features for model input.

3. **Model Training**  
   - Handling class imbalance using the scale_pos_weight parameter in XGBoost.  
   - Training an XGBoost classifier with tuned hyperparameters (`n_estimators=50`, `max_depth=3`).  

4. **Model Evaluation**  
   - Assessing performance using accuracy, confusion matrix, precision, recall, and F1-score.  
   - Observing good recall on the churn class, which is important for identifying customers likely to leave.

## Results

- Achieved approximately **75.9% accuracy** on the test set.  
- Demonstrated ability to effectively detect churners with a recall of **82%** on the minority class.  
- Precision for churn class was lower, suggesting further improvement potential.

## Future Work

- Hyperparameter tuning using Grid Search or Random Search to optimize model performance.  
- Applying advanced techniques like SMOTE or other resampling methods to better handle class imbalance.  
- Experimenting with other classification algorithms and ensemble methods for comparison.

## Requirements

- Python 3.x  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`

## Usage

1. Load and preprocess the dataset.  
2. Train the XGBoost classifier with specified parameters.  
3. Evaluate model predictions using various metrics.  
4. Visualize results through confusion matrices and feature correlations.


---

*This project demonstrates an end-to-end approach to customer churn prediction using a powerful machine learning technique, providing a foundation for further exploration and deployment.*
