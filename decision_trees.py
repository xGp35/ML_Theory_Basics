import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Any


data = pd.read_csv('data/iris.csv')
le = LabelEncoder()
data['type'] = le.fit_transform(data['species'])
data = data.drop(columns=['species'])

# col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
print(data.head())


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split = 2, max_depth = 2):
        """constructor"""

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        '''recursive function to build the tree'''
        X, Y = dataset[:, :-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        #split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            #find the best split
            





















