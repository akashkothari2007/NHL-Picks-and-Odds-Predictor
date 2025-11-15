import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load data
modelData = pd.read_csv("../data/processedData/finalModelTraining.csv")
modelData = modelData.fillna(0)

# Features and target
drop_cols = ['HOME_W', 'SEASON', 'GAME_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
X = modelData.drop(columns=drop_cols, errors='ignore')
y = modelData['HOME_W']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Best hyperparameters found from tuning
rf_best = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


#train and evaluate and save the model using pickle
rf_best.fit(X_train, y_train)

with open('nba_model.pkl', 'wb') as f:
    pickle.dump(rf_best, f)


rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
print(f"Tuned Random Forest: {rf_acc:.4f}")



