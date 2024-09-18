from sklearn_som.som import SOM
from sklearn import datasets

iris = datasets.load.iris()
iris_data = iris.data[:,:2]
iris_label = iris.target

iris_som = SOM(m=3,n=1,dim=2)
iris_som.fit(iris.data)

predictions = iris_som.predict(iris_data)

print(predictions)