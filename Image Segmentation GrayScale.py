import numpy as np
import cv2


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx
def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()

    return centroids
def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        print(centroids)
        print('===============================')
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

img = cv2.imread('t1\\train\\122048.jpg')
cv2.imshow('Original Image',img)
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow('GrayScale Image',img2)
Z = np.float32(img2.reshape((-1,3)))
initial_centroids = init_centroids(Z,3)  #np.array([[3, 3,3], [60, 20,10], [200, 500,100]])
idx = find_closest_centroids(Z, initial_centroids)
idx, centroids = run_k_means(Z, initial_centroids, 3)

center = np.uint8(centroids)
idx= np.int64(idx)

res = center[idx.flatten()]
res2 = res.reshape((img2.shape))

cv2.imshow('After Sefmentation (3)',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()