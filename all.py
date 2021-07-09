from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
import warnings
warnings.filterwarnings("ignore")  # 忽略版本问题

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re,string
# NLP
import nltk
from nltk.classify import ClassifierI
from statistics import mode #按众数表决

#建立分类器类
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
#遍历分类器对象列表。 然后，对于每一个，我们要求它基于特征分类。
    # 分类被视为投票。遍历完成后，返回mode(votes)，这只是返回投票的众数。
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
#置信度算法
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

df_data = pd.read_csv('/Users/lucong/PycharmProjects/AndroidorIOS/train.csv')


#查看训练集dataframe的数组规模
#print(df_data.shape)
#返回dataframe的前n行，默认n=5
#print(df_data.head())

#查看待预测集dataframe的数组规模
#print(df_valid.shape)
#print(df_valid.head())

#将Id，Score，ViewCount的平均值按IOS,ANDROID分组
#print(df_data.groupby(['LabelNum']).mean())

#查看train数据集的情况
#print(df_data.describe())

#nltk.download('punkt')                # this is a tokenizer
#nltk.download('wordnet')                    # lexical database (determine base word)
#nltk.download('averaged_perceptron_tagger'); # context of a word
#nltk.download('stopwords'); # stopwords

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)  #remove website name
        token = re.sub("(@[A-Za-z0-9_]+)","", token) # remove tagging of users
        token = re.sub("(<\/?\w*>)", "", token) # remove html
        #将所有词的词性限定为三种
        if tag.startswith("NN"):  #以NN开头通常是名词
            pos = 'n'
        elif tag.startswith('VB'):  #以VB开头通常是动词
            pos = 'v'
        else:
            pos = 'a'    #其余标注为a

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)  #词性标记，将每个单词都修改为词干 如：friends-friend

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# stopwords to be parsed into function `remove_noise` defined above
from nltk.corpus import stopwords
stop_words = stopwords.words('english') #获取英语停用词列表

all_tokens = df_data.apply(lambda row: nltk.word_tokenize(row['Title']), axis=1) #分词函数应用到"title"形成的一维数组上

cleaned_tokens = list()
for tokens in all_tokens:
    cleaned_tokens.append(remove_noise(tokens, stop_words))

df_data['cleaned_tokenized_titles'] = cleaned_tokens  #在训练集中新加一列"清洗后的titles"，列值=cleaned_tokens

df_data.cleaned_tokenized_titles[df_data.LabelNum == 0]


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


#from nltk import FreqDist

#freq_dist_apple = FreqDist(get_all_words(df_train.cleaned_tokenized_titles[df_data.LabelNum == 1].values))
#freq_dist_android = FreqDist(get_all_words(df_train.cleaned_tokenized_titles[df_data.LabelNum == 0].values))

#查看与IOS问题相关的最常见十大关键词
#print(freq_dist_apple.most_common(10))
#查看与Android问题相关的最常见十大关键词
#print(freq_dist_android.most_common(10))

#朴素贝叶斯

import random

def prep_tokens_for_model(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


# NLTK requires the data in this format:
android_data = [(title, 'Android') for title in prep_tokens_for_model(df_data.cleaned_tokenized_titles.values[df_data.LabelNum == 0])]
apple_data = [(title, 'Apple') for title in prep_tokens_for_model(df_data.cleaned_tokenized_titles[df_data.LabelNum == 1])]

data = android_data + apple_data
random.shuffle(data) #打乱数据集
train = data[:35959]
test = data[35959:]

x_test = [i[0] for i in train] # removing y_test, the correct label
y_test = [i[1] for i in train]  # saving y_test to evaluate the classifications

#classifier = NaiveBayesClassifier.train(train)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

y_preds = list()
for test1 in x_test:
    y_preds.append(classifier.classify(test1))

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier,test))*100)
print(confusion_matrix(y_test,y_preds))
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_clf = MNB_classifier.train(train)

y_preds = list()
for test2 in x_test:
    y_preds.append(MNB_clf.classify(test2))
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, test))*100)
print(confusion_matrix(y_test,y_preds))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_clf = BNB_classifier.train(train)
y_preds = list()
for test3 in x_test:
    y_preds.append(MNB_clf.classify(test3))
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, test))*100)
print(confusion_matrix(y_test,y_preds))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LR_clf = LogisticRegression_classifier.train(train)
y_preds = list()
for test4 in x_test:
    y_preds.append(LR_clf.classify(test4))
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LR_clf, test))*100)
print(confusion_matrix(y_test,y_preds))


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGD_clf = SGDClassifier_classifier.train(train)
y_preds = list()
for test5 in x_test:
    y_preds.append(SGD_clf.classify(test5))
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, test))*100)
print(confusion_matrix(y_test,y_preds))


voted_classifier = VoteClassifier(classifier,
                                  MNB_clf,
                                  BNB_clf,
                                  LR_clf,
                                  SGD_clf)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, test))*100)

print("Classification:", voted_classifier.classify(test[0][0]), "Confidence %:",voted_classifier.confidence(test[0][0])*100)
print("Classification:", voted_classifier.classify(test[1][0]), "Confidence %:",voted_classifier.confidence(test[1][0])*100)
print("Classification:", voted_classifier.classify(test[2][0]), "Confidence %:",voted_classifier.confidence(test[2][0])*100)
print("Classification:", voted_classifier.classify(test[3][0]), "Confidence %:",voted_classifier.confidence(test[3][0])*100)

