import pandas as pd
import seaborn as sns

df = sns.load_dataset("healthexp")  # or "tips", "diamonds", etc.
df = pd.get_dummies(df)

X = df.drop(['Life_Expectancy'], axis = 1)

y = df['Life_Expectancy']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19)

from sklearn.ensemble import RandomForestRegressor
#create model with params
rfr = RandomForestRegressor(n_estimators = 100, random_state = 13)
#training
rfr.fit(X_train, y_train)
#predicting
y_pred = rfr.predict(X_test)
print(y_pred)
print(y_test)

#evaluating
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))
print(r2_score(y_pred, y_test))

#grid of different params lots of combos
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#model that trains with all combination of params and finds best score
from sklearn.model_selection import GridSearchCV
rfr_cv = GridSearchCV(rfr, param_grid, cv = 10, scoring='neg_mean_squared_error', n_jobs = -1)

rfr_cv.fit(X_train, y_train)
y_pred = rfr_cv.predict(X_test)
print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))
print(r2_score(y_pred, y_test))
