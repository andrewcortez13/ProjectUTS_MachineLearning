import numpy as np
from sklearn.neighbors import KDTree
import pickle
#ref :https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

#1
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions #membuat 10 titik di 3d (total 30 data)

#melihat data dan modelnya
print (X)

tree = KDTree(X, leaf_size=2)   # tree yang kita buat menggunakan data dari X yang telah kita buat,dengan percabangan sebanyak 2      
dist, ind = tree.query(X[:1], k=3) #mencari 3 titik dari tree, dan distancenya
print ("k-nearest neighbour")               
print(ind)  # indices of 3 closest neighbors  #print 3 indices terdekat

print(dist)  # distances to 3 closest neighbors  #print 3 jarak terdekat


print ("pickle and unpickle tree")
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions x,y,z
tree = KDTree(X, leaf_size=2)        
s = pickle.dumps(tree)                     
tree_copy = pickle.loads(s)                
dist, ind = tree_copy.query(X[:1], k=3)     
print(ind)  # indices of 3 closest neighbors

print(dist)  # distances to 3 closest neighbors

#3 (neighbour in given radius)
print ("neighbour in given radius")
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)     
print(tree.query_radius(X[:1], r=0.3, count_only=True))

ind = tree.query_radius(X[:1], r=0.3)  
print(ind)  # indices of neighbors within distance 0.3 #indices dengan jarak range 0.3

#4 (Gaussian kernel density estimate)
print ("gaussian ")
rng = np.random.RandomState(42) 
X = rng.random_sample((100, 3))
tree = KDTree(X)                
#tree.kernel_density(X[:3], h=0.1, kernel='gaussian')
print (tree.kernel_density(X[:3], h=0.1, kernel='gaussian'))

#5 (two-point auto-correlation function)
# Untuk mencari korelasi 2 titik
print ("compute two point auto-correlation function")
rng = np.random.RandomState(0)
X = rng.random_sample((30, 3))
r = np.linspace(0, 1, 5)
tree = KDTree(X)                
#tree.two_point_correlation(X, r)
print (tree.two_point_correlation(X, r))