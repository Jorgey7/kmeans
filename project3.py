from __future__ import division, print_function, unicode_literals

import getopt
import hashlib
import sys
import os
import numpy
import pandas
import nltk

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer, pos_tag
from nltk import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


tf.keras.backend.clear_session()


def create_model(correct_labels, data_matrix, epochs):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(128, input_dim=len(data_matrix[0]), activation='relu'),
                                        tf.keras.layers.Dense(200, activation='softmax'),
                                        tf.keras.layers.Dense(256, input_dim=len(data_matrix[0]), activation='relu'),
                                        tf.keras.layers.Dense(128, activation='softmax'),
                                        tf.keras.layers.Dense(64, input_dim=len(data_matrix[0]), activation='relu'),
                                        tf.keras.layers.Dense(len(correct_labels[0]), activation='sigmoid')])
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
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

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
            if split_text[0] == 'ham':
                int_rep = 0
            else:
                int_rep = 1
            labels.append(int_rep)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text)

    better_vectors = pandas.DataFrame(vectors.toarray().transpose(), index=vectorizer.get_feature_names())
    better_vectors = better_vectors.transpose()

    numpy_labels = numpy.array(labels, dtype=int)
    data = numpy.array(better_vectors, dtype=float)
    return numpy_labels, data


def create_matrix_csv(filename, group_col):
    matrix = numpy.genfromtxt(filename, delimiter=',', dtype=str)
    labels = numpy.array(matrix[:, [group_col]]).flatten()
    int_labels = numpy.zeros_like(labels)
    for i in range(0, len(labels)):
        temp_str = str(labels[i])
        int_rep = int(hashlib.md5(temp_str.encode('utf-8')).hexdigest(), 16)
        int_labels[i] = (int_rep % 10 ** 2)

    data_matrix = numpy.delete(matrix, [group_col], axis=1).astype(float, order='K')
    print(labels.dtype)
    return int_labels, data_matrix


def main(argv):
    inputfile = ""
    group_col = 0
    file_type = ""
    epochs = 10

    try:
        opts, args = getopt.getopt(argv, "hi:g:k:t:")
    except getopt.GetoptError:
        print('project3 -i <inputfile> -g <column of group in csv data> -t <type (sms or csv)>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('project3 -i <inputfile> -g <column of group in csv data> -t <type (sms or '
                  'csv)>')
            sys.exit()
        elif opt == '-i':
            inputfile = arg
            assert(os.path.exists(inputfile))
        elif opt == '-g':
            group_col = int(arg)
        elif opt == '-t':
            file_type = arg

    if file_type == "csv":
        (correct_labels, data_matrix) = create_matrix_csv(filename=inputfile, group_col=group_col)
    elif file_type == "sms":
        (correct_labels, data_matrix) = import_sms(filename=inputfile)
    else:
        sys.exit(-1)

    processed_labels = tf.keras.utils.to_categorical(correct_labels)
    results = create_model(processed_labels, data_matrix, epochs)
    print(results)


if __name__ == "__main__":
    main(sys.argv[1:])
