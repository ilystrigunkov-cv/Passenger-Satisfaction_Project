# Airline Passenger Satisfaction: Exploratory Data Analysis

Exploratory data analysis of the [Airline Passenger Satisfaction Dataset](https://www.kaggle.com/competitions/playground-series-s6e10/data) (~700,000 flight records), with a machine learning model to predict passenger satisfaction.

## Goal

Identify which factors have the greatest influence on passenger satisfaction — and predict satisfaction levels using a machine learning model.

## What's inside

* **Data cleaning** — handling missing values in `Arrival Delay in Minutes`
* **Data transformation** — renaming long column names for easier access and coding
* **Visualization** — correlation heatmap to identify relationships between numerical features and the target variable
* **Machine learning** — CatBoostClassifier trained on 13 features (including 4 categorical features handled natively), achieving **0.956 ROC AUC** on passenger satisfaction prediction

## Key findings

* `Online boarding`, `Inflight entertainment`, and `Seat comfort` are the strongest numerical predictors of satisfaction (correlation 0.56, 0.43, and 0.40 respectively)
* Flight delays (`DepDelMin`, `ArDelMin`) show a surprisingly low direct linear correlation with overall satisfaction (-0.04 and -0.05)
* The dataset is naturally well-balanced (approximately 56% dissatisfied vs 44% satisfied), meaning complex class-balancing techniques are not required for the baseline model
