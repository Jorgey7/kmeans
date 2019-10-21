from __future__ import division, print_function, unicode_literals

import getopt
import sys
import os
import numpy
import pandas

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer, pos_tag
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


tf.keras.backend.clear_session()


def create_model(correct_labels, data_matrix, epochs):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, input_dim=len(data_matrix[0]), activation='relu'),
                                        tf.keras.layers.Dropout(.2),
                                        tf.keras.layers.Dense(64, activation='relu'),
                                        tf.keras.layers.Dense(len(correct_labels), activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_size = int(len(data_matrix) * .8)
    training_data = data_matrix[:training_size]
    training_labels = correct_labels[:training_size]
    testing_data = data_matrix[training_size:]
    testing_labels = correct_labels[training_size:]

    print()
    model.fit(x=training_data, y=training_labels, epochs=epochs)
    results = model.evaluate(x=testing_data, y=testing_labels)
    del model
    return results


def import_sms(filename):
    # nltk.download('wordnet')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')

    k = 2
    lemmatizer = WordNetLemmatizer()
    token = RegexpTokenizer(r'\w+')
    tag_dict = {"J": wordnet.wordnet.ADJ,
                "N": wordnet.wordnet.NOUN,
                "V": wordnet.wordnet.VERB,
                "R": wordnet.wordnet.ADV}

    stopword = set(stopwords.words('english'))

    with open(filename, 'r') as file:
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

    numpy_labels = numpy.array(labels, order='K', dtype='str')
    data = numpy.array(better_vectors, dtype=float)
    return numpy_labels, data


def create_matrix_csv(filename, group_col):
    matrix = numpy.genfromtxt(filename, delimiter=',', dtype=str)
    labels = numpy.array(matrix[:, [group_col]]).flatten()
    data_matrix = numpy.delete(matrix, [group_col], axis=1).astype(float, order='K')
    print(labels.dtype)
    return labels, data_matrix


def main(argv):
    inputfile = ""
    group_col = 0
    file_type = ""
    epochs = 10
    k_value = -1

    try:
        opts, args = getopt.getopt(argv, "hi:g:k:t:")
    except getopt.GetoptError:
        print('kmeans -i <inputfile> -g <column of group in csv data> -t <type (sms or csv)>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('kmeans -i <inputfile> -g <column of group in csv data> -t <type (sms or '
                  'csv)>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
            assert(os.path.exists(inputfile))
        elif opt == '-g':
            group_col = int(arg)
        elif opt == '-k':
            k_value = int(arg)
        elif opt == '-t':
            file_type = arg

    if file_type == "csv":
        (correct_labels, data_matrix) = create_matrix_csv(filename=inputfile, group_col=group_col)
        print(type(correct_labels))
        print(correct_labels.shape)
        print(type(data_matrix))
        print(data_matrix.shape)
    elif file_type == "sms":
        (correct_labels, data_matrix) = import_sms(filename=inputfile)
        # correct_labels = numpy.array(correct_labels_temp)
        print(type(correct_labels))
        print(type(data_matrix))
        print(correct_labels.shape)
        print(data_matrix.shape)
    else:
        sys.exit(-1)

    # correct_labels =
    if k_value != -1:
        results = create_model(correct_labels, data_matrix, epochs, k_value)
    print(results)


if __name__ == "__main__":
    main(sys.argv[1:])
