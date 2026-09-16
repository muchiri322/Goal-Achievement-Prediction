import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Features: [sleep_hours, water_glasses, bench_kg]
X = np.array([
    [7.5, 7, 80], [8.0, 8, 82], [6.5, 6, 78], [7.0, 9, 85],
    [9.0, 8, 80], [7.5, 7, 83], [8.0, 8, 84], [6.0, 6, 81],
    [8.5, 9, 85], [7.0, 8, 80], [7.5, 8, 86], [9.0, 7, 79],
    [7.0, 9, 84], [7.5, 8, 83], [7.0, 7, 82], [8.0, 8, 86],
    [6.5, 6, 79], [7.5, 9, 88], [8.0, 8, 81], [7.0, 7, 85],
    [8.5, 9, 87], [7.0, 8, 82], [7.5, 8, 86], [6.5, 6, 80],
    [8.0, 9, 87], [9.5, 7, 79], [7.0, 8, 84], [8.0, 9, 86]
])

# Target from the project notebook:
# 1 = goal hit (10,000+ steps)
# 0 = below goal
steps = np.array([
    9200, 10500, 8800, 11000, 7600, 9400, 10200, 8900,
    10800, 9100, 11200, 7900, 10000, 9700, 9500, 10300,
    8600, 11500, 8200, 9800, 10600, 9000, 10100, 8400,
    10900, 7500, 9600, 10400
])

y = (steps >= 10000).astype(int)


def train_model():
    """Train the Random Forest model and save it to model.joblib."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Below goal", "Goal hit"],
        zero_division=0
    ))

    joblib.dump(model, "model.joblib")
    print("\nModel saved to model.joblib")

    return model


def predict_day(model, sleep_hours, water_glasses, bench_kg):
    """Predict whether a day is likely to hit the 10,000-step goal."""
    inputs = np.array([[sleep_hours, water_glasses, bench_kg]])

    prediction = model.predict(inputs)[0]
    probabilities = model.predict_proba(inputs)[0]
    confidence = max(probabilities) * 100

    hit_goal = bool(prediction)

    if hit_goal:
        recommendation = "High step day likely. Good conditions today."
    else:
        recommendation = "Low step day likely. Schedule a walk this afternoon."

    return {
        "hit_goal": hit_goal,
        "label": "Goal hit" if hit_goal else "Below goal",
        "confidence": round(confidence, 1),
        "recommendation": recommendation
    }


if __name__ == "__main__":
    model = train_model()

    scenarios = [
        (8.0, 8, 84),
        (6.0, 5, 78),
        (9.0, 9, 87),
        (7.0, 7, 82),
    ]

    print("\nPredictions:")
    for sleep, water, bench in scenarios:
        result = predict_day(model, sleep, water, bench)

        print(
            f"sleep={sleep}h | water={water} glasses | "
            f"bench={bench}kg => {result['label']} "
            f"({result['confidence']}% confidence)"
        )
        print(f"  {result['recommendation']}")
