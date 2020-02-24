from sklearn.ensemble import RandomForestClassifier
import numpy as np
x_train = np.array([
    [1, 2],
    [3, 4],
    [-1, 2],
    [-3, 4]
])

y_train = [1, 1, 0, 0]
clf_rf = RandomForestClassifier()
clf_rf.fit(x_train, y_train)
 
print(clf_rf.predict([[2, 2]]))  # [1] это предсказанный класс
print(clf_rf.predict_proba([[2, 2]]))  # [[0.2 0.8]] вероятности по классам
