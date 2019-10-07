import time
import getopt
import os
import sys
from math import sqrt

import numpy


def l2_distances(data_one, data_two):
    dist_total = 0
    for dim in range(0, len(data_one)):
        dist_total += (data_one[dim] - data_two[dim])**2
    return sqrt(dist_total)


def calculate_confusion(centers, correct_labels, labels, k):
    my_dict = {}
    for i in range(len(labels)):
        if (correct_labels[i], labels[i]) in my_dict:
            my_dict[correct_labels[i], labels[i]] = my_dict[correct_labels[i], labels[i]] + 1
        else:
            my_dict[correct_labels[i], labels[i]] = 1
    print(my_dict)
    total = 0
    correct = [0] * k
    for value in my_dict.values():
        for count in range(k):
            if value > correct[count]:
                correct[count] = value
                break
        total += value
    correct_sum = 0
    for count in range(k):
        correct_sum += correct[count]
    print("precision: {}".format(correct_sum/total))
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
    new_ind = numpy.random.RandomState()
    i = new_ind.permutation(matrix.shape[0])[:k]
    centers = matrix[i]
    print(type(centers))

    while(True):
        labels = distances_argmin(matrix, centers)
        new_centers = numpy.empty_like(centers)
        for i in range(len(centers)):   # len(centers) aka i is equal to k
            mylist = []
            for j in range(len(labels)):    # j is equal to matrix row index
                if labels[j] == i:
                    mylist.append(j)
            new_centers[i] = sum(matrix[mylist])/len(mylist)
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
    start = time.time()
    (centers, new_labels) = kMeans(data_matrix, k_value, iterations)
    calculate_confusion(centers, correct_labels, new_labels, k_value)
    print(time.time() - start)
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
