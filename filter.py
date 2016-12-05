from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
 
stoplist = stopwords.words('english')
 
def init_lists(folder):
    key_list = []
    file_content = os.listdir(folder)
    for a_file in file_content:
        f = open(folder + a_file, 'r')
        key_list.append(f.read())
    f.close()
    return key_list
 
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]
 
def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}
 
def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set of size= ' + str(len(train_set)) + ' mails')
    print ('Test set of size = ' + str(len(test_set)) + ' mails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier
 
def evaluate(train_set, test_set, classifier):
    # test accuracy of classifier on training and test set
    print ('Training set accuracy = ' + str(classify.accuracy(classifier, train_set)))
    print ('Test set accuracy = ' + str(classify.accuracy(classifier, test_set)))
    # check most informative words for the classifier
    classifier.show_most_informative_features(20)
 
if __name__ == &amp;quot;__main__&amp;quot;:
    # initialise the data
    spam = init_lists('enron1/spam/')
    ham = init_lists('enron1/ham/')
    all_mails = [(mail, 'spam') for mail in spam]
    all_mails += [(mail, 'ham') for mail in ham]
    random.shuffle(all_mails)
    print ('Corpus of size = ' + str(len(all_mails)) + ' mails')
 
    # extract the features
    all_features = [(get_features(mail, ''), label) for (mail, label) in all_mails]
    print ('Fetched ' + str(len(all_features)) + ' feature sets')
 
    # train the classifier
    train_set, test_set, classifier = train(all_features, 0.8)
 
    # evaluate performance
    evaluate(train_set, test_set, classifier)
