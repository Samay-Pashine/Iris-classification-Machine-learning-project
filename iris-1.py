import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)
dataframe = pd.DataFrame(x_train, columns=dataset.feature_names)
pd.plotting.scatter_matrix(dataframe, c= y_train, marker='o')
A = KNeighborsClassifier(n_neighbors=1)
A.fit(x_train, y_train)
sl, sw, pl, pw = map(float, input("Enter the Sepal length, width and Petal length, width (in order) : ").split()) #5, 2.9, 1, 0.2
New = np.array([[sl, sw, pl, pw]])
delta = A.predict(New)
print("The Species of Delta is : ", dataset['target_names'][delta])
print("Accuracy percentage : ", A.score(x_test, y_test)*100, "%")