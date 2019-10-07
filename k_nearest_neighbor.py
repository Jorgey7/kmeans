import sys
import numpy
import nltk
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer, pos_tag
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def calculate_confusion(centers, correct_labels, labels, k):
    my_dict = {}
    for i in range(len(labels)):
        if (correct_labels[i], labels[i]) in my_dict:
            my_dict[correct_labels[i], labels[i]] = my_dict[correct_labels[i], labels[i]] + 1
        else:
            my_dict[correct_labels[i], labels[i]] = 1
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
    print("Kmeans accuracy: {}".format(correct_sum/total))
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


def kMeans(matrix, k):
    new_ind = numpy.random.RandomState()
    i = new_ind.permutation(matrix.shape[0])[:k]
    centers = matrix[i]

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


def l2_distances(data_one, data_two):
    return numpy.sqrt(((data_one - data_two)**2).sum(axis=0))


def k_nearest_neighbor(vectors, labels, k):
    temp = []

    for data in vectors:
        distance = []
        for other_data in vectors:
            distance.append(l2_distances(data, other_data))
        close = []
        for x in sorted(zip(distance, labels)):
            close.append(x)

        temp.append(max(set(close[0:k]), key=close.count))

    accuracy = 0
    for i in range(len(labels)):
        if labels[i] in temp[i]:
            accuracy += 1

    print("KNN accuracy: {}".format(accuracy/len(labels)))
    return


def main():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    print("Advisory: Run will take approximately 5 minutes to complete!")

    k = 2
    lemmatizer = WordNetLemmatizer()
    token = RegexpTokenizer(r'\w+')
    tag_dict = {"J": wordnet.wordnet.ADJ,
                "N": wordnet.wordnet.NOUN,
                "V": wordnet.wordnet.VERB,
                "R": wordnet.wordnet.ADV}

    stopword = set(stopwords.words('english'))

    with open('data/smsspamcollection/SMSSpamCollection', 'r') as file:
        text = file.readlines()
        all_text = []
        labels = []
        for i, j in enumerate(text):
            split_text = j.rsplit('\t')
            temp = split_text[1]
            tokens = token.tokenize(temp)
            new_tokens = [w for w in tokens if not w in stopword]
            tokens = new_tokens
            lemmatized_output = ' '.join([lemmatizer.lemmatize(w, tag_dict.get(tag[0], wordnet.wordnet.NOUN))
                                         for w, tag in pos_tag(tokens)])
            all_text.append(lemmatized_output)
            labels.append(split_text[0])

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text)

    better_vectors = pandas.DataFrame(vectors.toarray().transpose(), index=vectorizer.get_feature_names())
    better_vectors = better_vectors.transpose()

    data = numpy.array(better_vectors)

    (centers, new_labels) = kMeans(data, k)
    calculate_confusion(centers, labels, new_labels, k)

    k_nearest_neighbor(data, labels, k)


if __name__ == "__main__":
    main()
