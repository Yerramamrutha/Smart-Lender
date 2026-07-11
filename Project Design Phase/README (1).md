# Project Development Phase

This folder documents progress across each epic of the project lifecycle.

## Epic 1: Data Collection and Architecture Design
- Dataset downloaded and stored in the `Dataset/` folder.
- Application architecture defined (see `Project Design Phase/README.md`).

## Epic 2: Visualizing and Analyzing the Data
- Performed univariate, bivariate, and multivariate analysis using Pandas,
  Matplotlib, and Seaborn to understand feature distributions and
  relationships (see `train_model.py` in `Project Executable Files/`).

## Epic 3: Data Pre-Processing
- Handled missing values, removed duplicates, encoded categorical variables,
  balanced the target classes using SMOTE, scaled numerical features, and
  split the data into training and test sets.

## Epic 4: Model Building
- Trained and evaluated four models: Decision Tree, Random Forest, KNN, and
  XGBoost. XGBoost achieved the best test accuracy and was selected for
  deployment.

## Epic 5: Application Building
- Built an HTML front-end form and a Flask backend that loads the trained
  model and returns real-time loan approval predictions.
