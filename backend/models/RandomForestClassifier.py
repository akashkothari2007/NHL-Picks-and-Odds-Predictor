import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier


def trainModel():
    # Load data
    modelData = pd.read_csv("data/processedData/finalModelTraining.csv").fillna(0)

    # Features + target
    drop_cols = ['HOME_W', 'SEASON', 'GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    X = modelData.drop(columns=drop_cols, errors='ignore')
    y = modelData['HOME_W']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Best params 
    best_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 2,
        "subsample": 0.8
    }

    model = GradientBoostingClassifier(**best_params)

    # Train
    model.fit(X_train, y_train)

    # Save model
    with open('models/nba_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTuned GradientBoosting Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")


def predictSingleGame(df):
    # Load model
    with open("models/nba_model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(df)
    prob_matrix = model.predict_proba(df)

    probabilities = []
    for pred, prob in zip(predictions, prob_matrix):
        probabilities.append(prob[1] if pred == 1 else prob[0])

    return list(predictions), probabilities