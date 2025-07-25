# Customer Churn Prediction and Analysis

A comprehensive machine learning project for predicting customer churn using advanced analytics and multiple modeling approaches.

## Project Overview

This project implements a complete machine learning pipeline for customer churn prediction, from data collection to model deployment. The solution leverages multiple advanced machine learning models and provides a [user-friendly web interface](https://huggingface.co/spaces/mohamedmostafa259/SmartChurnPredictor) for predictions.

### Dataset 

The **Customer Churn Risk Rate** dataset contains structured tabular data, including, information about customer demographics, account activity, transactional behavior, and other customer-related features that can be leveraged for churn prediction. It was originally provided as part of a HackerEarth Machine Learning Challenge aimed at predicting customer churn risk.  

**Sources:**

- **Kaggle**: [Churn Risk Rate - HackerEarth ML](https://www.kaggle.com/datasets/imsparsh/churn-risk-rate-hackerearth-ml/data)  
- **HackerEarth Competition**: [Predict Customer Churn](https://www.hackerearth.com/challenges/new/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/)

## Project Structure

The project is organized into four main milestones:

### Milestone 1: Data Collection, EDA, and Data Cleaning

- Data collection and initial exploration
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing pipelines

### Milestone 2: Feature Engineering and Advanced Analysis

- Feature engineering implementation
- Advanced statistical analysis (hypothesis testing, customer segmentation, correlation analysis)
- Data transformation pipelines

### Milestone 3: Model Development

- Implementation of multiple machine learning models: LogisticRegression (baseline), RandomForest, SVC, XGBoost, CatBoost, and LightGBM
- Hyperparameter tuning for promising models: XGBoost, CatBoost, and LightGBM
- Error Analysis (for class 2 and 4)
- Custom model with adjusted predicted probabilites
- Full ML pipeline (used in deployment)

### Milestone 4: Model Deployment

- Web application development using Gradio

![app_screenshot1](https://github.com/MohamedMostafa259/Customer-Churn-Prediction-and-Analysis/blob/main/visuals/app_screenshot1.png?raw=true)
![app_screenshot1](https://github.com/MohamedMostafa259/Customer-Churn-Prediction-and-Analysis/blob/main/visuals/app_screenshot2.png?raw=true)

## Project Organization

Each milestone contains:
- Jupyter notebooks for analysis and development
- Python scripts for pipeline implementation
- Documentation (to be added)

## Results

- The best xgboost model achieved f1_macro score of **0.78** which strongly competes with the score of the 1st winner of the competition made on this dataset. [He built a model with **0.77** f1_macro](https://www.hackerearth.com/challenges/new/competitive/hackerearth-machine-learning-challenge-predict-customer-churn/#:~:text=Machine%20Learning%20practice-,Winners,-Adarsh%20Wase).
- The project successfully implements a customer churn prediction system.
- During error analysis and exploratory modeling, I discovered that the original churn scores (1–5) exhibit two distinct behavioral groupings: 
    - Scores 1 & 2: Customers showing similar behavior patterns and low churn risk.
    - Scores 3, 4 & 5: Customers with higher churn tendencies.
    ![days_since_last_login_by_churn_score](https://github.com/MohamedMostafa259/Customer-Churn-Prediction-and-Analysis/blob/main/visuals/days_since_last_login_by_churn_score.png?raw=true)
    ![t-sne_segmentation](https://github.com/MohamedMostafa259/Customer-Churn-Prediction-and-Analysis/blob/main/visuals/t-sne_segmentation.png?raw=true)


## Future Improvements

Potential areas for enhancement:
- PowerBI dashboard
- Readme file for each milestone
- Milestone5: Final Documentation and Presentation, for stakeholders, that showcases the project's results and business impact.
- Adding some visualizations in and input examples in the web application interface (using Gradio on HuggingFace)
