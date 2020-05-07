# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:02:47 2020

@author: Ashima
"""

import time
import config
import utils
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


class Model():

    def __init__(self):
        self.classifier = None

    def class_accuracy(self, cmatrix, string):
        print("Accuracy of each class in {} Set".format(string))
        cmatrix_sum = np.sum(cmatrix, axis=1)
        for i in range(config.CLASSES):
            acc = cmatrix[i][i]/cmatrix_sum[i]
            print("Accuracy of class {} {}".format(i, acc))

    def svc_train(self, data, net):
        X_train, Y_train = data.load_features(data, net)
        print("No of examples to train: ", X_train.shape[0])
        start_time = time.time()
        self.classifier = SVC(gamma='auto', probability=True)
        self.classifier.fit(X_train, Y_train)
        end_time = time.time()
        print("Training time: {}".format(end_time - start_time))
        print("Train Accuracy: ", self.classifier.score(X_train, Y_train))
        y_predict = self.classifier.predict(X_train)
        y_predict_proba = self.classifier.predict_proba(X_train)
        print("y_predict_proba train: ", y_predict_proba.shape)
        print("y_predict train: ", y_predict.shape)
        print("Confusion Matrix of Train Set")
        cm = confusion_matrix(Y_train, y_predict)
        print(cm)
        precision = precision_score(Y_train, y_predict, average="weighted")
        print("Weighted Precision Score: ", precision)
        recall = recall_score(Y_train, y_predict, average="weighted")
        print("Weighted Recall Score: ", recall)
        f1 = f1_score(Y_train, y_predict, average="weighted")
        print("Weighted F1 Score: ", f1)
        utils.plot_roc(Y_train, y_predict_proba, "Train")

    def svc_test(self, data, net):
        X_test, Y_test = data.load_features(data, net)
        print("Test accuracy: ", self.classifier.score(X_test, Y_test))
        y_predict = self.classifier.predict(X_test)
        y_predict_proba = self.classifier.predict_proba(X_test)
        print("Confusion Matrix of Test Set")
        cm = confusion_matrix(Y_test, y_predict)
        print(cm)
        precision = precision_score(Y_test, y_predict, average="weighted")
        print("Weighted Precision Score: ", precision)
        recall = recall_score(Y_test, y_predict, average="weighted")
        print("Weighted Recall Score: ", recall)
        f1 = f1_score(Y_test, y_predict, average="weighted")
        print("Weighted F1 Score: ", f1)
        utils.plot_roc(Y_test, y_predict_proba, "Test")
