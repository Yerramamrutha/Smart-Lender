# Project Design Phase

This folder contains the design artifacts created before development began.

## Contents
- `ER_Diagram.png` — Entity Relationship Diagram showing the database schema:
  `USER`, `APPLICANT_PROFILE`, `CREDIT_HISTORY`, `LOAN_APPLICATION`, `MODEL`,
  and `PREDICTION_RESULT` tables and how they relate to each other.

## Architecture Summary
Data flow: `Applicant Data → Preprocessing → Trained ML Model → Prediction
Result → Stored & Displayed via Flask Web App`

The system is designed for deployment on IBM Cloud, with Flask serving as
the web layer connecting the front-end form to the trained XGBoost model.
