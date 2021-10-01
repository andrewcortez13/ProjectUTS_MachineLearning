#library yang digunakan
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#libary untuk evaluasi
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

#mengenerate data menggunakan function make_blobs, data yang dibuat 200,dengan 3 center, stdev = 2.75, 
features, true_labels = make_blobs(
    n_samples=200,
    centers=3,
    cluster_std=2.75,
    random_state=42
)
#dari make_blobs function , membuat tuple dengan 2 isi yang berbeda  
    #isi 1 : 2d array numpy array dengan x, y untuk setiap sample yang dibuat
    #isi 2 : 1d array numpy array dengan cluster label untuk setiap sample yang ada

#hasil dari make_blobs, mengambil array 0-4 untuk di print, isi 1
print ("features : " )
print (features[:5])

#hasil isi 2, numpy array dan clusternya, diambil array 0-4
print ("true labels : ")
print (true_labels[:5])

#agar dapat berjalan lebih baik di machine learning, dilakukan scaling, data prespocessing
#penting dilakukan sebelum machine learning dijalankan
#scaling disini menjadikan semua data mean =0, stdev =1
scaler = StandardScaler()

#print data setelah di scaling
print ("scaled features")

#setelah di scaling, data sudah siap diclustering (dipetakan)
scaled_features = scaler.fit_transform(features)
print (scaled_features[:5])

#clustering k-means
kmeans = KMeans(
    init="random", #mengatur mode ke random, menyetel ke k-means++, metode ini untuk mempercepat data bertemu dalam satu titik
    n_clusters=3, #n cluster/k cluster diset 3
    n_init=10, #mengatur inisialisasi 10,penting karena dua run, dapat berkumpul pada tugas yang berbeda, biasanya algoritma sklearn
    #menjalankan 10 k-means, dan return SSE terendah
    max_iter=300, #membatasi batas iterasi per k-means
    random_state=42
)

#data, dimasukkan ke scaled feature
kmeans.fit(scaled_features) #fit data scaled_features, melakukan 10x algoritma k-means pada data, max 300 iterasi per proses

#statistik data dari inisialisasi SSE terendah, masuk ke kmeans setelah di fit()
#lihat intertia, center
print ("kmeans inertia")
print (kmeans.inertia_)
# lokasi tengah kmeans cluster
print ("kmeans cluster centers")
print (kmeans.cluster_centers_)
# jumlah iterasi yang diperlukan agar kmeans menyatu
print ("kmeans N-iter")
print (kmeans.n_iter_)

# penetapan cluster, 1d array, prediksi 5 arr pertama
print ("5 predicted labels")
print (kmeans.labels_[:5])

#elbow methods, cara ini menjalankan beberapa k-means, dan naikkan k setiap iterasi, dan mencatat SSE nya
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# list untuk menampung SSE dari setiap iterasi k
sse = []

#iterasi 10x, dan kmeans k ^ kmeans_kwargs, lalu kmeans inertia 
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

#plot graph sse dengan number of clusters
#ketika kita plot SSE sebagai function dari angka cluster, SSE berkurang setiap kali bertambah, lalu data akan mendekat
#plt untuk show graph dari matplotlib
plt.title("clustering k means")
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

#mencari elbow method, result = 3
kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
kl.elbow

#cara menggunakan silhoutte coefficients
#silhoute coeficience mengukur seberapa dekat titik dengan titik lain di dalam metode cluster. Koefisien
#bernilai -1 dan 1, lebih besar angka, berarti lebih dekat
#siluet rata-rata dijadikan 1 score, fungsi silluet_score membutuhkan minimal 2 cluster/ akan menimbulkan pengecualian
#tidak menghitung SSEnya, tapi menghitung koefisien siluet

# list untuk tempat silhoutte koefisien untuk setiap k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.title("clustering silhoutte")
plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# Untuk evaluasi performa clustering
features, true_labels = make_moons(
    n_samples=250, noise=0.05, random_state=42
)
scaled_features = scaler.fit_transform(features)

# Buat instance k-means dan dbscan algoritm
kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)

# Sesuaikan algoritma dengan fitur
kmeans.fit(scaled_features)
dbscan.fit(scaled_features)

# hitung skor siloute untuk setiap algoritma
kmeans_silhouette = silhouette_score(
    scaled_features, kmeans.labels_
).round(2)
dbscan_silhouette = silhouette_score(
   scaled_features, dbscan.labels_
).round (2)

kmeans_silhouette
dbscan_silhouette
# Plot data dan perbandingan silluet cluster
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8, 6), sharex=True, sharey=True
)
fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
fte_colors = {
    0: "#008fd5",
    1: "#fc4f30",
}
# K-Means plot
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
ax1.set_title(
    f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
)

# dbscan plot
db_colors = [fte_colors[label] for label in dbscan.labels_]
ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
ax2.set_title(
    f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
)
plt.show()
