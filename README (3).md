# Goal Achievement Prediction

A small machine learning project that predicts whether a day is likely to **hit a 10,000-step goal** using three inputs:

- Sleep hours
- Water intake in glasses
- Bench press weight in kilograms

The project uses a **Random Forest Classifier** and also returns a prediction confidence and a simple recommendation.

## Project Structure

```text
goal-prediction/
│
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
└── model.joblib        # created after running model.py
```

## How It Works

The model uses:

```text
Features
--------
sleep_hours
water_glasses
bench_kg

Target
------
1 = Goal hit
0 = Below goal
```

The original project data defines a goal hit as achieving **10,000 or more steps**.

The model is trained using a Random Forest classifier.

## Example

For a day with:

```text
Sleep: 8 hours
Water: 8 glasses
Bench: 84 kg
```

the model can return a result such as:

```text
Goal hit (high confidence)
```

The exact confidence depends on the trained model.

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd goal-prediction
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python model.py
```

This will:

1. Load the training data.
2. Create the target based on the 10,000-step threshold.
3. Split the data into training and testing sets.
4. Train a Random Forest classifier.
5. Display test accuracy and a classification report.
6. Save the trained model as `model.joblib`.
7. Run several example predictions.

## Model Function

The main prediction function is:

```python
predict_day(model, sleep_hours, water_glasses, bench_kg)
```

Example:

```python
result = predict_day(model, 8.0, 8, 84)

print(result)
```

Example output:

```text
{
    'hit_goal': True,
    'label': 'Goal hit',
    'confidence': 85.0,
    'recommendation': 'High step day likely. Good conditions today.'
}
```

## Important Note

This is a small demonstration machine learning project built from a limited sample dataset. The model should be treated as a learning/development example rather than a scientifically validated fitness or health prediction system.

The three input variables are used to predict the project's defined step-goal outcome; they do not establish that sleep, hydration, or bench press independently cause the result.

## Technologies

- Python
- NumPy
- Scikit-learn
- Random Forest
- Joblib

## Future Improvements

- Collect a larger real-world dataset.
- Add daily step history.
- Add additional activity and lifestyle features.
- Build a Flask or FastAPI prediction API.
- Create a web frontend.
- Store predictions in a database.
- Track prediction performance over time.

## Author

Muchiri

Data Engineer | Data Analyst | AI/ML
