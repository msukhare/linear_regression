# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_lineair.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/05/30 13:44:17 by msukhare          #+#    #+#              #
#    Updated: 2018/06/01 17:04:53 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys
import csv

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("error name file")
    data.insert(0, '0', 1)
    col = data.shape[1]
    X = data.iloc[:, 0:col-1]
    Y = data.iloc[:, col-1:]
    X = np.array(X.values, dtype=float)
    Y = np.array(Y.values, dtype=float)
    thetas = np.zeros(((col - 1), 1), dtype=float) 
    tmp_thetas = np.zeros(((col - 1), 1), dtype=float) 
    return (data, X, Y, thetas, tmp_thetas)

def get_min_max(X, j, row, min_or_max):
    i = 0
    to_ret = X[i][j]
    while (i < row):
        if (min_or_max == 1 and to_ret > X[i][j]):
            to_ret = X[i][j]
        elif (min_or_max == 0 and to_ret < X[i][j]):
            to_ret = X[i][j]
        i += 1
    return (to_ret)

def scale_feat(X_scale, j, row, min_x, max_x):
    i = 0
    while (i < row):
        X_scale[i][j] = X_scale[i][j] / (max_x - min_x)
        i += 1

def scale_X(col_X, row_X, X_scale, X):
    j = 1
    while (j < col_X):
        min_X = get_min_max(X, j, row_X, 1)
        max_X = get_min_max(X, j, row_X, 0)
        scale_feat(X_scale, j, row_X, min_X, max_X)
        j += 1

def hypo(X_scale, i, thetas, col_X):
    j = 0
    to_ret = 0
    while (j < col_X):
        to_ret += thetas[j][0] * X_scale[i][j]
        j += 1
    return (to_ret)

def cost_fct(thetas, X_scale, Y, row_X, col_X):
    i = 0
    somme = 0
    while (i < row_X):
        somme += (hypo(X_scale, i, thetas, col_X) - Y[i][0])**2
        i += 1
    return ((1 / (2 * row_X)) * somme)

def get_sum(X_scale, Y, row_X, col_X, pl_the, thetas):
    i = 0
    somme = 0
    while (i < row_X):
        somme += (hypo(X_scale, i, thetas, col_X) - Y[i][0]) * X_scale[i][pl_the]
        i += 1
    return (somme)

def guardient_descent(thetas, tmp_thetas, X_scale, Y, row_X, col_X):
    i = 0
    alpha = 0.1
    while (i < col_X):
        tmp_thetas[i][0] = thetas[i][0] - (alpha * (1 / row_X) * get_sum(X_scale, Y, row_X, col_X, i, thetas))
        i += 1

def make_predi(thetas, tmp_thetas, X_scale, Y, row_X, col_X):
    res_bef = 0
    while (res_bef != cost_fct(thetas, X_scale, Y, row_X, col_X)):
        res_bef = cost_fct(thetas, X_scale, Y, row_X, col_X)
        guardient_descent(thetas, tmp_thetas, X_scale, Y, row_X, col_X)
        i = 0
        while (i < col_X):
            thetas[i][0] = tmp_thetas[i][0]
            i += 1

def write_in_file(thetas, col_X):
    try:
        c = csv.writer(open(sys.argv[2], "w"))
    except:
        sys.exit("fail to creat file")
    i = 0
    while (i < col_X):
        c.writerow([str(thetas[i][0])])
        i += 1
    

def main():
    data, X, Y, thetas, tmp_thetas = read_file()
    X_scale = np.array(X, dtype=float)
    row_X = X.shape[0]
    col_X = X.shape[1]
    scale_X(col_X, row_X, X_scale, X)
    make_predi(thetas, tmp_thetas, X_scale, Y, row_X, col_X)
    j = 1
    while (j < col_X):
        min_X = get_min_max(X, j, row_X, 1)
        max_X = get_min_max(X, j, row_X, 0)
        thetas[j][0] = thetas[j][0] / (max_X - min_X)
        j += 1
    write_in_file(thetas, col_X)

if __name__ == "__main__":
    main()
