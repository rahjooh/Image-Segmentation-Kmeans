import numpy as np
k = 5
centroids = [1,2,3,4,5]
idx = [1,2,3,4,5]
X = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
for i in range(k):
    indices = np.where(idx == i)
    print(indices)
    centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()