# Smart-Lender
Smart Lender: ML-powered web app that predicts loan approval using Decision Tree, Random Forest, KNN, and XGBoost, deployed with Flask.
# Smart Lender - Loan Approval Prediction

Smart Lender is a machine learning-powered web application that predicts the
creditworthiness of loan applicants, helping banks and financial institutions
make faster, data-driven loan approval decisions.

## Overview
The platform evaluates applicant data (income, credit history, employment,
education, etc.) using four classification algorithms — Decision Tree, Random
Forest, K-Nearest Neighbors (KNN), and XGBoost — and deploys the best
performing model (XGBoost) inside a Flask web application for real-time
predictions.

## Tech Stack
- Python, Flask
- NumPy, Pandas
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- HTML/CSS

## Repository Structure
| Folder | Contents |
|---|---|
| `Project Design Phase/` | ER diagram and system architecture |
| `Project Development Phase/` | Development progress and epic-wise breakdown |
| `Project Executable Files/` | Source code — model training script and Flask app |
| `Dataset/` | Loan applicant dataset used for training |
| `Documents/` | Project report / documentation |
| `Project Demo & Video/` | Demo video link |

## How to Run
1. `cd "Project Executable Files"`
2. `pip install -r requirements.txt`
3. `python train_model.py` (trains models, saves best one)
4. `python app.py` (starts Flask server)
5. Open `http://127.0.0.1:5000`

## Project Workflow (Epics)
1. **Data Collection and Architecture Design**
2. **Visualizing and Analyzing the Data**
3. **Data Pre-Processing**
4. **Model Building**
5. **Application Building**
