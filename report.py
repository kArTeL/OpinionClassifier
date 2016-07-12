
from __future__ import print_function
from itertools import chain
from utility import get_document, open_pickled_file, show_most_informative_features, get_classification
import io
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import nltk
import pickle


encode = 'latin1'

print("Loading featuresets_all")
featuresets = open_pickled_file("featuresets_all.pickle")

#amount = len(documents)/10
#training_len = amount*9
#print("Training len", int(training_len))

# set that we'll train our classifier with
training_set = featuresets[:]

# set that we'll test against.
#testing_set = featuresets[:433] + featuresets[866:]

print("Loading NaiveBayes model")

f = open('classifier.pickle')
classifier = pickle.load(f)
f.close()

print("Finished loading model")
#
#
# print("Loading word_features all")
word_features = open_pickled_file("word_features_all.pickle")
#
# classifier = nltk.NaiveBayesClassifier

def find_features(document):
    words = document.lower().strip().split(' ')
    features = {}
    for w in word_features:
        features[w] = (w.lower().decode(encode) in [x.lower().decode(encode) for x in words])

    return features

def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)


def sentiment_file(file_path):
    #fo = io.open("result_file.csv", "w+", encoding=encode)

    file_text = ""
    document = get_document(file_path)
    expectedArray = []
    classifiedArray = []
    for index in range(len(document[0])):
        print("Processing %d of %d..." % (index + 1, len(document[0])))
        text = document[0][index].split(',')[2].strip().lower()
        classification = document[0][index].split(',')[1].strip()

        if (classification != "NONE"):
            classification = "OPINION"
        ##Insert to the expecte array
        expectedArray.append(classification)

        classified = sentiment(text)
        #file_text += "%s,%s\n" % (document[0][index].decode(encode), classified.upper())

        ## Append the classified
        classifiedArray.append(classified)
        #print("added in array expected: %s and classified as: %s" % (classification,classified))
    return bio_classification_report(expectedArray,classifiedArray)


def bio_classification_report(y_true, y_pred):
    print ("y_true")
    print(y_true)

    print ("y_predict")
    print (y_pred)
    """
    Classification report for a list of BIO-encoded sequences.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    print("y_true size: %d and y_pred size: %d" % (len(y_true), len(y_pred)))
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(y_true)
    y_pred_combined = lb.transform(y_pred)

    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    print("tagset:")
    print(tagset)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    print("indices:")
    print(class_indices)

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

print(sentiment_file("Test.csv"))
