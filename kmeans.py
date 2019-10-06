import getopt
import os
import sys
from math import sqrt

import numpy
import random


def l2_distances(data_one, data_two):
    dist_total = 0
    for dim in range(0, len(data_one)):
        dist_total += (data_one[dim] - data_two[dim])**2
    return sqrt(dist_total)


def calculate_confusion(centers, correct_labels, labels, k):
    # for label in correct_labels:
    #     print(label)
    # for data in centers:
    #     print(data)
    print(centers)
    # new_label = numpy.zeros_like(centers)
    # for i in range(k):
    #     mask = (centers == i)
    #     new_label[mask] = mode(correct_labels[mask])[0]
    # accuracy_score(correct_labels, new_label)

    return None


def distances_argmin(matrix, centers):
    label = []
    for i in range(0, len(matrix)):
        min_distance = sys.maxsize
        group = -1
        for j in range(len(centers)):
            distance = l2_distances(matrix[i], centers[j])
            if distance < min_distance:
                min_distance = distance
                group = j
        label.append(group)
    return label


def kMeans(matrix, k, iterations):
    seed = random.randrange(0, 100000)
    new_ind = numpy.random.RandomState(seed)
    i = new_ind.permutation(matrix.shape[0])[:k]
    centers = matrix[i]
    print(type(centers))

    # centers = numpy.empty((0))
    # for i in range(0, k):
    #     my_random = random.randrange(0, len(matrix))
    #     centers = numpy.append(centers, matrix[my_random])

    while(True):
        labels = distances_argmin(matrix, centers)
        # new_centers = numpy.array([matrix[labels == i].mean(0) for j in range(k)])
        new_centers = []
        for i in range(len(centers)):
            for j in range(len(labels)):
                if labels[j] == i:
                    pass    # todo
        print(type(new_centers))
        if numpy.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


def create_matrix(filename, group_col):
    matrix = numpy.genfromtxt(filename, delimiter=',', dtype=str)
    labels = numpy.array(matrix[:, [group_col]]).flatten()
    data_matrix = numpy.delete(matrix, [group_col], axis=1).astype(float, order='K')
    return labels, data_matrix


def main(argv):
    inputfile = ""
    k_value = 0
    group_col = 0
    iterations = 1

    try:
        opts, args = getopt.getopt(argv, "hi:k:g:")
    except getopt.GetoptError:
        print('kmeans -i <inputfile> -k <number of clusters> -g <column of group in csv data>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('kmeans -i <inputfile> -k <number of clusters> -g <column of group in csv data>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
            assert(os.path.exists(inputfile))
        elif opt == '-k':
            k_value = int(arg)
        elif opt == '-g':
            group_col = int(arg)

    print(inputfile)
    print(k_value)
    print(group_col)

    (correct_labels, data_matrix) = create_matrix(filename=inputfile, group_col=group_col)
    (centers, new_labels) = kMeans(data_matrix, k_value, iterations)
    calculate_confusion(centers, correct_labels, new_labels, k_value)

    return None


if __name__ == "__main__":
    main(sys.argv[1:])
