import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# зчитуємо дані та отримуємо розміри та класи
data_set = pd.read_csv("Glass.csv")
glass_size = data_set.iloc[:, :-1].values
glass_class = data_set.iloc[:, 9].values

# виконуємо нормалізацію числових ознак
scaler = MinMaxScaler()
scaler_glass_size = scaler.fit_transform(glass_size)

# кластеризація, алгоритм K-means, евклідова відстань
clusterer = KMeans(n_clusters=6)
clusterer.fit(scaler_glass_size)
labels = clusterer.labels_
metrics.silhouette_score(scaler_glass_size, labels, metric="euclidean")

# розділяємо скло по кластерам
predictions = clusterer.predict(scaler_glass_size)

# додаємо отримані результати розподілу у таблицю
data_set["cluster"] = predictions
print("Results:")
print(data_set, "\n")

# ПОБУДОВА ГРАФІКІВ
# отримання центроїдів
centroids = clusterer.cluster_centers_
print("Координати усіх центроїдів: ")
print(centroids, "\n")
#метод побудови графіка
def build_graphic(index1, index2):
    fig, ax = plt.subplots()
    scatter1 = ax.scatter(scaler_glass_size[:, index1], scaler_glass_size[:, index2],
                          c=predictions, s=15, cmap="brg")
    handles, labels = scatter1.legend_elements()
    legend1 = ax.legend(handles, labels, loc="upper right")
    ax.add_artist(legend1)
    scatter2 = ax.scatter(centroids[:, index1], centroids[:, index2], marker="x",
                          c="purple", s=200, linewidth=3, label="centroids")
    plt.legend(loc="lower right")
    plt.xlabel(f"{data_set.columns[index1]} after scaling")
    plt.ylabel(f"{data_set.columns[index2]} after scaling")
    plt.show()
# графік 1(пара ознак: RI - Na)
build_graphic(0, 1)
# графік 2(пара ознак: Mg - Al)
build_graphic(2, 3)
# графік 3(пара ознак: Si - K)
build_graphic(4, 5)
# графік 4(пара ознак: Ca - Ba)
build_graphic(6, 7)

# загальна кількість скла у кожному із кластерів
count_cluster = Counter(labels)
print("Number of objects in each cluster: ")
print(count_cluster, "\n")

# визначимо скільки у кожному кластері об'єктів кожного з класів
cluster_content = data_set.groupby(["cluster", "Type"]).size().unstack(fill_value=0)
cluster_content["Total"] = cluster_content.sum(axis=1)
cluster_content.loc["Total"] = cluster_content.sum()
print(tabulate(cluster_content, headers="keys", tablefmt="psql"))

# визначення оптимальної кількості кластерів
df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
for i in range(2, 11):
    clusterer_i = KMeans(n_clusters=i).fit(scaler_glass_size)
    predictions_i = clusterer_i.predict(scaler_glass_size)

    # сума квадратів відстаней від екземплярів до найближчого центроїда (WCSS)
    WCSS = clusterer_i.inertia_

    # Silhouette Score(силует)
    Silhouette = metrics.silhouette_score(scaler_glass_size, predictions_i)

    # Davies-Boudin Score
    DB = metrics.davies_bouldin_score(scaler_glass_size, predictions_i)

    new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
    df = pd.concat([df, new_row_df], ignore_index=True)
print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))

# ПОБУДОВА ГРАФІКІВ МЕТОДІВ
def build_method_graphic(method_name):
    plt.plot(df["Number of clusters"], df[method_name], marker="o", linestyle="None", label=method_name)
    plt.xlabel("Number of clusters")
    plt.ylabel(method_name)
    plt.title(method_name + " method")
    plt.legend()
    plt.show()
# метод ліктя
build_method_graphic("WCSS")
# метод силуету
build_method_graphic("Silhouette")
# індекс девіса-булдіна
build_method_graphic("DB")

# Вплив масштабування на кластеризацію та визначення оптимальної кількості кластерів
clusterer = KMeans(n_clusters=6)
clusterer.fit(glass_size)
labels = clusterer.labels_
metrics.silhouette_score(glass_size, labels, metric="euclidean")
predictions = clusterer.predict(glass_size)
data_set["cluster"] = predictions
print("Результати кластеризації без масштабування: ")
print(data_set, "\n")
count_cluster = Counter(labels)
print("Кількість у кластерах: ")
print(count_cluster, "\n")
cluster_content = data_set.groupby(["cluster", "Type"]).size().unstack(fill_value=0)
cluster_content["Total"] = cluster_content.sum(axis=1)
cluster_content.loc["Total"] = cluster_content.sum()
print(tabulate(cluster_content, headers="keys", tablefmt="psql"))
df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
for i in range(2, 11):
    clusterer_i = KMeans(n_clusters=i).fit(scaler_glass_size)
    predictions_i = clusterer_i.predict(scaler_glass_size)

    # сума квадратів відстаней від екземплярів до найближчого центроїда (WCSS)
    WCSS = clusterer_i.inertia_

    # Silhouette Score(силует)
    Silhouette = metrics.silhouette_score(scaler_glass_size, predictions_i)

    # Davies-Boudin Score
    DB = metrics.davies_bouldin_score(scaler_glass_size, predictions_i)

    new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
    df = pd.concat([df, new_row_df], ignore_index=True)
print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))
build_method_graphic("WCSS")
# метод силуету
build_method_graphic("Silhouette")
# індекс девіса-булдіна
build_method_graphic("DB")









