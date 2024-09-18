import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

#lectura de los datos iris
iris = load_iris()
iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
iris_df ["target"] = iris.target
iris_df.head()

#carga del metodo k means
kmeans = KMeans(n_clusters=3, init= "k-means++" ,max_iter= 300, random_state = 0) 
kmeans.fit(iris.data)
print(kmeans.labels_)

#datos de graficacion de cluster
iris_df["cluster"] = kmeans.labels_
iris_df.groupby (["target","cluster"]).agg({'sepal length (cm)' : 'count'})

#graficacion
pca  = PCA(2)

pca_res = pca.fit_transform(iris.data)
iris_df['x'] = pca_res [:, 0]
iris_df['y'] = pca_res [:, 1]
iris_df.head()

#asignacion de clusteres
cluster_0 = iris_df[iris_df['cluster'] == 0]
cluster_1 = iris_df[iris_df['cluster'] == 1]
cluster_2 = iris_df[iris_df['cluster'] == 2]

plt.scatter(cluster_0['x'], cluster_0['y'], label = "Cluster_0")
plt.scatter(cluster_1['x'], cluster_1['y'], label = "Cluster_1")
plt.scatter(cluster_2['x'], cluster_2['y'], label = "Cluster_2")

#leyendas
plt.legend()
plt.title('')
plt.xlabel('x')
plt.ylabel('y')
plt.show()