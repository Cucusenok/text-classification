from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
 
 
 
x_train = [
    [1, 2], [5, 6],
    [3, 4], [7, 8],
    [-1, 2], [-5, 6],
    [-3, 4], [-7, 8], [0, 0]
]
y_train = [1, 1, 1, 1, 0, 0, 0, 0, 1]
 
parameter_grid = {
            'criterion': ['entropy', 'gini'],
            'max_depth': [10, 20, 100],
            'n_estimators': [10, 20, 100]
        }
clf_rf = RandomForestClassifier()
clf_rf.fit(x_train, y_train)
 
importances = clf_rf.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
 
indices = np.argsort(importances)[::-1]
names_indices = ['x_coor', 'y_coor']
 
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
 
plt.bar(range(len(importances)), importances[indices], color="r")
plt.xticks(range(len(importances)), names_indices, rotation=90)
 
plt.tight_layout()
plt.xlim([-1, len(importances)])
plt.show()