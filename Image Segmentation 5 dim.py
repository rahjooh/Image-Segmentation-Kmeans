import numpy as np
import scipy as sp
from scipy import ndimage
import cv2
def find_closest_centroids(X, centroids):
    m1 = X.shape[0]
    m2 = X.shape[1]
    k1 = centroids.shape[0]
    idx = np.zeros((m1,m2))

    for row in range(m1):
        for col in range(m2) :
            min_dist = 1000000
            for q in range(k1):
                dist = np.sum(([row,col,X[row,col,0],X[row,col,1],X[row,col,2]] - centroids[q,:]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    idx[row,col] = q
    return idx
def compute_centroids(X, idx, k):
    m, n ,p = X.shape
    centroids = np.zeros((k, p+2))
    for i in range(k):

        for t1 in range(idx.shape[0]):
            for t2 in range(idx.shape[1]):
                indices = np.where(idx == i)
                if(idx[t1][t2] == i):
                    print("'",end='')
                    for q in range(len(centroids[i, :])):
                        centroids[i, q] = (np.sum(X[t1,t2,q], axis=1) / len(indices[0]))

        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    return centroids

def run_k_means(X, initial_centroids, max_iters):
    m, n ,p= X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        #print(centroids)
        #print('===============================')
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids
def init_centroids(X, k):
    m, n ,p= X.shape
    centroids = np.zeros((k, p+2))
    idx = np.random.randint(0, m, k)
    idy= np.random.randint(1,n,k)
    for i in range(k):
        centroids[i,:] = [i**2,n-i**3,X[idx[i],idy[i],[0]],X[idx[i],idy[i],[1]],X[idx[i],idy[i],[2]]]
    return centroids


t= sp.ndimage.imread('t1\\train\\2092.jpg', flatten=False, mode=None)
initial_centroids = init_centroids(t , 5)
idx = find_closest_centroids(t, initial_centroids)
#print(idx)
print(idx.shape)
print(initial_centroids.shape)
idx, centroids = run_k_means(t, initial_centroids, 3)






"""
img = cv2.imread('t1\\train\\353013.jpg')
cv2.imshow('Original Image',img)
Z = np.float32(img.reshape((-1,3)))
initial_centroids = init_centroids(Z,5)  #np.array([[3, 3,3], [60, 20,10], [200, 500,100]])
idx = find_closest_centroids(Z, initial_centroids)
idx, centroids = run_k_means(Z, initial_centroids, 5)

center = np.uint8(centroids)
idx= np.int64(idx)

res = center[idx.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('After Sefmentation (5)',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""