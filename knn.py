# -*- coding: utf-8 -*-
"""
@author: Jack
"""
#from scipy.spatial import distance
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
    def fit(self, x_train, y_train):
        #Doesn't really do anything other than save x and y train, as well as
        #making sure that they are arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.x_train = x_train
        self.y_train = y_train
    def predict(self, test):
        test = np.array(test)
        #instantiating an empty list to be used later
        alloftest = []
        
        #itterates through the x_test array
        for i in range(len(test)):
            distances = []
            for index in range(len(self.x_train)):
                #calculate distance between the test and all the training sets
                #currently a place holder
                dist = np.linalg.norm(test[i] - self.x_train[index])
                #appends the distance for use later
                distances.append((self.x_train[index], dist, self.y_train[index]))
            
            #sorts the distances and appends the K closest distances
            distances.sort(key=lambda x: x[1])
            alloftest.append(distances[:self.k])
         
        output = []    
        for i in range(len(alloftest)):
            # I discovered Counter while researching KNN. It's a handy way to
            # keep tallies and then choose which one is the most common
            class_counter = Counter()
            for boi in alloftest[i]:
                # Finds out how many times it's closes to each possibility
                class_counter[boi[2]] += 1
                
            # Appends the one with the most
            output.append(class_counter.most_common(1)[0][0])
            
        return(output)