#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:06 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos Categóricos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Codificar datos categoricos X
from sklearn import preprocessing
LabelEncoder_X = preprocessing.LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float64)

# Codificar datos categoricos y
LabelEncoder_y = preprocessing.LabelEncoder()
y = LabelEncoder_y.fit_transform(y)