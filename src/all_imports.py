# coding=utf-8
import json
import math
import os
import os.path
import random as rn
import tempfile
import time
import warnings
from collections import Counter
from functools import wraps
from math import sqrt
from pdb import set_trace
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.decomposition as dc
#import tabulate
from matplotlib import pyplot as plt
from numpy import std
from pylab import rcParams
from sklearn import metrics, model_selection, svm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_squared_error,
                             precision_recall_curve,
                             precision_recall_fscore_support, precision_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, ParameterGrid,
                                     RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#from tabulate import _table_formats, tabulate
from yellowbrick.cluster import KElbowVisualizer

# import keras
#import smote_variants as sv
#import tensorflow as tf
from data_reader import GetData
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.callbacks import EarlyStopping
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras_balanced import KerasIMB
# from keras_weights import Tensor_Class
from kkerasclass import K_Class
from lightgbm import LGBMClassifier
from llgbm import L_LGBM
from pca import PCA1
#from personality import Personality
#from plot_confusion_matrix import *
from preprocess import Preprocess
from randomf import RandomForest
#from tensorflow import keras

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""  Custom imports """



#from process import GetData