import numpy as np
import pandas as pd
from pre_process_tab import load_preprocess
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle


class LROracle():
    def __init__(self):
        self.lr= LogisticRegression(random_state=42)
    def train(self, train_x, train_y):
        self.lr.fit(train_x, train_y)
    def oracle_pred(self, x,y, evaluate=False):
        pred=self.lr.predict(x)
        if evaluate:
            total= x.shape[0]
            correct= (pred==y).sum()
            print('lr oracle accuracy:', correct/total)
        return pred

class DecisionTree():
    def __init__(self):
        #self.tree= DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
        self.tree= DecisionTreeClassifier(random_state=42, class_weight='balanced')
    def train(self, train_x, train_y):
        self.tree.fit(train_x, train_y)
    def oracle_pred(self, x,y, evaluate=False):
        pred=self.tree.predict(x)
        if evaluate:
            total= x.shape[0]
            correct= (pred==y).sum()
            print('lr oracle accuracy:', correct/total)
        return pred

class RandomForest():
    def __init__(self):
        self.rf=RandomForestClassifier(class_weight='balanced', random_state=42)
        #self.rf= RandomForestClassifier(max_depth=3,class_weight='balanced', random_state=42)
    def train(self, train_x, train_y):
        self.rf.fit(train_x, train_y)
    def oracle_pred(self, x,y, evaluate=False):
        pred=self.rf.predict(x)
        if evaluate:
            total= x.shape[0]
            correct= (pred==y).sum()
            print('lr oracle accuracy:', correct/total)
        return pred

class SVM():
    def __init__(self):
        self.svm= SVC()
    def train(self, train_x, train_y):
        self.svm.fit(train_x, train_y)
    def oracle_pred(self, x,y, evaluate=False):
        pred=self.svm.predict(x)
        if evaluate:
            total= x.shape[0]
            correct= (pred==y).sum()
            print('lr oracle accuracy:', correct/total)
        return pred

class LGB():
    def __init__(self):
        self.gb= LGBMClassifier(max_depth=3, n_estimators=100, learning_rate=0.01,class_weight='balanced')
    def train(self, train_x, train_y):
        self.gb.fit(train_x, train_y)
    def oracle_pred(self, x,y, evaluate=False):
        pred=self.gb.predict(x)
        if evaluate:
            total= x.shape[0]
            correct= (pred==y).sum()
            print('lr oracle accuracy:', correct/total)
        return pred


if __name__=='__main__':
    TARGET= 'over_50k'
    train_x, test_x, train_y, test_y= load_preprocess("/home/hadrien/data/adult/adult.csv", ['fnlwgt'], TARGET)
    train_x, valid_x, train_y, valid_y= train_test_split(train_x, train_y, train_size=.8, random_state=42)
    lr= LROracle()
    lr.train(train_x, train_y)
    with open("./models/lr", "wb") as fp:   #Pickling
        pickle.dump(lr, fp)
    tree= DecisionTree()
    tree.train(train_x, train_y)
    with open("./models/tree", "wb") as fp:   #Pickling
        pickle.dump(tree, fp)
    rf= RandomForest()
    rf.train(train_x, train_y)
    with open("./models/rf", "wb") as fp:   #Pickling
        pickle.dump(rf, fp)
    svm=SVM()
    svm.train(train_x, train_y)
    with open("./models/svm", "wb") as fp:   #Pickling
        pickle.dump(svm, fp)
    gb=LGB()
    gb.train(train_x, train_y)
    with open("./models/gb", "wb") as fp:   #Pickling
        pickle.dump(gb, fp)



    
