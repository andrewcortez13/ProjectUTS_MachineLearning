import numpy as np
from sklearn.neighbors import KDTree
import pickle
#ref :https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

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

####################################################################
#ref : https://github.com/Vectorized/Python-KD-Tree/blob/master/kdtree.py
"""
A super short KD-Tree for points...
so concise that you can copypasta into your homework without arousing suspicion.
Usage:
1. Use make_kd_tree to create the kd
2. You can then use `get_knn` for k nearest neighbors or 
   `get_nearest` for the nearest neighbor
points are be a list of points: [[0, 1, 2], [12.3, 4.5, 2.3], ...]
"""
# Makes the KD-Tree for fast lookup
# Membuat KD-Tree untuk pencarian cepat
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i), #rekursi membuat kd_tree dari setengah data
            make_kd_tree(points[half + 1:], dim, i), #rekursi setengah sisanya
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]

# Adds a point to the kd-tree
# menambahkan titik point di kd-tree
def add_point(kd_node, point, dim, i=0):
    if kd_node is not None:
        dx = kd_node[2][i] - point[i]
        i = (i + 1) % dim
        for j, c in ((0, dx >= 0), (1, dx < 0)):
            if c and kd_node[j] is None:
                kd_node[j] = [None, None, point]
            elif c:
                add_point(kd_node[j], point, dim, i)

# k nearest neighbors
#function untuk get_k nearest neighbour
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]

# For the closest neighbor
# Function untuk neighbor terdekat
def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        # Di taruh di sisi kiri, dan kanan jika diperlukan
        for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
            get_nearest(kd_node[b], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]

"""
If you want to attach other properties to your points, 
you can use this class or subclass it.
Usage:
point = PointContainer([1,2,3])
point.label = True  
print point         # [1,2,3]
print point.label   # True 
"""
class PointContainer(list):
    def __new__(self, value, name = None, values = None):
        s = super(PointContainer, self).__new__(self, value)
        return s


"""
Below is all the testing code
"""

import random, cProfile


def puts(l):
    for x in l:
        print(x)


def get_knn_naive(points, point, k, dist_func, return_distances=True):
    neighbors = []
    for i, pp in enumerate(points):
        dist = dist_func(point, pp)
        neighbors.append((dist, pp))
    neighbors = sorted(neighbors)[:k]
    return neighbors if return_distances else [n[1] for n in neighbors]

dim = 3

#random.uniform() adalah function dari numpy untuk merandom angka secara uniform
def rand_point(dim):
    return [random.uniform(-1, 1) for d in range(dim)]

#perhitungan distance
def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in range(dim)) 

def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)


points = [PointContainer(rand_point(dim)) for x in range(10000)] #pembuatan total points 10k points, disini angka random
#points yang diambil distribusi beragam, pake library numpy(function rand_point) di line 116
#random uniform : https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html

additional_points = [PointContainer(rand_point(dim)) for x in range(50)]
#points = [rand_point(dim) for x in range(5000)]
test = [rand_point(dim) for x in range(100)]
result1 = []
result2 = []

#membuat function bench1, dengan membuat 10k sample data, dengan 3 titik dim
def bench1():
    kd_tree = make_kd_tree(points, dim)
    for point in additional_points: 
        add_point(kd_tree, point, dim)
    result1.append(tuple(get_knn(kd_tree, [0] * dim, 8, dim, dist_sq_dim))) 
    print (kd_tree )
    for t in test:
        result1.append(tuple(get_knn(kd_tree, t, 8, dim, dist_sq_dim)))


def bench2():
    all_points = points + additional_points
    result2.append(tuple(get_knn_naive(all_points, [0] * dim, 8, dist_sq_dim)))
    for t in test:
        result2.append(tuple(get_knn_naive(all_points, t, 8, dist_sq_dim)))

cProfile.run("bench1()")
cProfile.run("bench2()")

puts(result1[0])
print("")
puts(result2[0])
print("")

print("Is the result same as naive version?: {}".format(result1 == result2))

kd_tree = make_kd_tree(points, dim) #membuat kd_tree dengan function di atas dengan 10k points, dan 3 dim

print(get_nearest(kd_tree, [0] * dim, dim, dist_sq_dim)) #check yang terdekat menggunakan kd_tree

"""
You can also define the distance function inline, like:
print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: dist_sq(a, b, dim))
print get_nearest(kd_tree, [0] * dim, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in range(dim)))
"""