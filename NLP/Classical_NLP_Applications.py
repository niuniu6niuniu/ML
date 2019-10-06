# - - -   1. Name Gender Classifier   - -  - #
# code to build a classifier to classify names as male or female
# demonstrates the basics of feature extraction and model building
import nltk

names = [(name, 'male') for name in nltk.corpus.names.words("male.txt")]
names += [(name, 'female') for name in nltk.corpus.names.words("female.txt")]

def extract_gender_features(name):
    name = name.lower()
    features = {}
    # define features
    features["suffix"] = name[-1:]
    features["suffix2"] = name[-2:] if len(name) > 1 else name[0]
    features["suffix3"] = name[-3:] if len(name) > 3 else name[0]
    features["prefix"] = name[:1]
    features["prefix2"] = name[:2] if len(name) > 1 else name[0]
    features["prefix3"] = name[:3] if len(name) > 2 else name[0]
    features["prefix4"] = name[:4] if len(name) > 3 else name[0]
    features["prefix5"] = name[:5] if len(name) > 4 else name[0]
    features["wordLen"] = len(name)
    return features

# data = [(extract_gender_features(name), gender) for (name, gender) in names]

# import random
# random.shuffle(data)
# print(data[:10])
# print()
# print(data[-10:])

# dataCount = len(data)
# trainCount = int(.8*dataCount)
# # - Split data into train and test
# trainData = data[:trainCount]
# testData = data[trainCount:]
# bayes = nltk.NaiveBayesClassifier.train(trainData)

def classify(name):
    label = bayes.classify(extract_gender_features(name))
    print("name =", name, " classified as =", label)

# print("trainData accuracy=", nltk.classify.accuracy(bayes, trainData))
# print("testData accuracy=", nltk.classify.accuracy(bayes, testData))
# - Test some samples
# classify("david")
# classify("susan")
# classify("alex")
# - Show most informative features
# bayes.show_most_informative_features(25)

# Check the errors
# errors = []
#
# for (name, label) in names:
#     if bayes.classify(extract_gender_features(name)) != label:
#         errors.append({"name": name, "label": label})

# print(errors)


# - - -   2. Sentiment Analysis   - - - #
from nltk.corpus import movie_reviews as reviews
import random

docs = [(list(reviews.words(id)), cat) for cat in reviews.categories() for id in reviews.fileids(cat)]
random.shuffle(docs)
# print([(len(d[0]), d[0][:2], d[1]) for d in docs[:10]])
fd = nltk.FreqDist(word.lower() for word in reviews.words())
topKeys = [ key for (key, value) in fd.most_common(2000)]

# import nltk
def review_features(doc):
    docSet = set(doc)
    features = {}

    # Bag of Words model
    for word in topKeys:
        features[word] = (word in docSet)

    return features

# print(review_features(reviews.words("pos/cv957_8737.txt")))

data = [(review_features(doc), label) for (doc, label) in docs]

dataCount = len(data)
trainCount = int(.8*dataCount)

trainData = data[:trainCount]
testData = data[trainCount:]

bayes2 = nltk.NaiveBayesClassifier.train(trainData)

print("trainData accuracy=", nltk.classify.accuracy(bayes2, trainData))
print("testData accuracy=", nltk.classify.accuracy(bayes2, testData))

bayes2.show_most_informative_features(20)