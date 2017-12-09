'''look at all these friendly helper methods
'''
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def cohesion(cluster_dict):
    # we're not scientists so just return mean cohesion for each cluster
    clust_means = []
    for k, v in cluster_dict:
        if k is not 'Noise':
            points = v
            dists = []
            for p1 in points:
                for p2 in points:
                    if p1 is not p2:
                        dists.append(np.linalg.norm(p1 - p2))

            clust_means.append(np.mean(dists))

    return np.mean(clust_means)


def separation(cluster_dict):
    dists = []
    for key in cluster_dict.keys():
        if key is not 'Noise':
            mean_point = np.mean(cluster_dict[key], axis=0)
            for key2 in cluster_dict.keys():
                if key2 is not 'Noise' and key is not key2:
                    mean_point2 = np.mean(cluster_dict[key2], axis=0)
                    dists.append(distance(mean_point, mean_point2))
    return np.mean(dists)


def report():
    pass


def crossVal():
    pass

# parse in a dataset from the data folder by name
# argStart index is where to start slicing a line to capture all args
# argEnd index is were to stop slicing args
# class is the index where the class label is located
# example for zoo dataset
# x, y = helpers.readDatasetFromFile('zoo.data', 1, -1, -1)
# zoo dataset has a non-feature in the first position so start looking at 
# index 1. features continue until the last index, which is the class label
# output is a tuple (x, y) where x is a matrix of input vectors, y is a matrix of output vectors
def read_data(filename, argStartIndex, argEndIndex, classIndex):
    outs = {}
    with open('../data/' + filename, 'r') as f:
        for line in f:
            linearr = line.split(',')
            attrs = tuple(map(float, linearr[argStartIndex: argEndIndex]))
            label = linearr[classIndex]

            outs[attrs] = (label, None)  # label should always be index 0 in tuple, cluster next
    return outs


def readGlassData():
    return read_data('glass.data', 1, 10, 10)  # need shape [9 ... 7]


def readWineData():
    return read_data('wine.data', 1, 14, 0)  # need shape [13 ... 3]


def readLeafData():
    return read_data('leaf.csv', 2, 16, 0)  # need shape [14 ... 30]


def read_some_other_data():
    pass


def read_iris_data():
    return read_data('iris.data', 0, 5, 5)


def distance(a, b):
    return np.linalg.norm(a - b)

# if you need the inputs as an np matrix use this
def get_input_matrix_from_dict(read_data_output):
    return np.vstack(map(np.array, read_data_output.keys()))

#moving from dictionary key to list of floats       
def str_to_float(lst):
    x_L=re.findall(r"[-+]?\d*\.\d+|\d+", lst)
    x_floats= [float(i) for i in x_L]
    return x_floats

#moving from float to string to use as key in dictionary
def float_to_str(lst):
    x=''
    for i in lst:
        x += (repr(i))
    return x

#using knn to educate the tuning process for min_pts and eps in DBSCAN
#para x is the result of get_input_matrix_from_dict()
def knn_plot(x):
    means =[]
    #for all possible k, run knn
    for i in range(1,len(x)):
        neigh = NearestNeighbors(n_neighbors=i).fit(x)
        distances, indices = neigh.kneighbors(x)
        means.append(distances.mean())
    #plot the mean distance between points at each k value  
    plt.plot(means)
    plt.show()
    