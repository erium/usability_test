import os
import shutil
from distutils.dir_util import copy_tree

import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from joblib import dump, load




def load_data(path):

    example_df = datasets.load_diabetes(return_X_y=False, as_frame=True)['frame'] # The default example of the template - sklearn diabetes
    if path == 'default example':
        df = example_df
    else:
        df = pd.read_csv(path)
    
    return df


alpha_param = np.logspace(-4, 0, num=50) 

poly_limit = 3 # Limit for polynomial degree, default = 4
max_depth = None # Limit for decision tree/random forest depth, default = None

#Possible models: ['linear', 'l1_linear', 'l2_linear', 'poly', 'tree', 'forest', 'mlp']
run_models = ['linear', 'poly']#, 'tree', ]#'l1_linear', 'l2_linear'] # Specify the models you would like to run in the list

