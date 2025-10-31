#IRIS DATASET TEST
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data #samples of irises with four columns (petal length ,width etc)
y = iris.target #target labels

#20% set aside for training of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#random state sets random seed so its consistent across multiple runs

#100 trees in the forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state = 42)

#training
rf_classifier.fit(X_train, y_train)

#Make predictions
y_pred = rf_classifier.predict(X_test)
print(y_pred)
print(y_test)

#evaluations

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred) #diagonal elements should be nonzero if right, everuthing else should be 0
print(result)

result1 = classification_report(y_test, y_pred)
print(result1)

result2 = accuracy_score(y_test, y_pred)
print(result2)

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
tree.plot_tree(rf_classifier.estimators_[0], feature_names=iris.feature_names, filled=True)
plt.show()