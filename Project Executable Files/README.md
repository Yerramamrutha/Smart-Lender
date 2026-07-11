# Project Executable Files

Source code for training the model and running the web application.

## Files
- `train_model.py` — Loads the dataset, performs EDA, preprocesses data,
  trains Decision Tree / Random Forest / KNN / XGBoost models, and saves
  the best model as `model.pkl`.
- `app.py` — Flask application that loads `model.pkl` and serves
  predictions through a web form.
- `templates/index.html` — Front-end form for entering applicant details.
- `requirements.txt` — Python dependencies.

## How to Run
```bash
pip install -r requirements.txt
python train_model.py   # trains models and saves model.pkl
python app.py            # starts the Flask server
```
Then open `http://127.0.0.1:5000` in your browser.

