# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_lineair.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/05/30 13:44:17 by msukhare          #+#    #+#              #
#    Updated: 2018/05/31 17:11:13 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import sys

def read_file():
    [file_name] = sys.argv[1:]
    try:
        data = pd.read_csv(file_name)
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

def hypo(X_scale, i, thetas, col_X):
    j = 0
    to_ret = 0
    while (j < col_X):
        to_ret += thetas[j][0] * X_scale
        j += 1
    return (to_ret)

def squar(hypo, y):
    return (hypo * hypo - (2 * hypo * y) + (y * y))

def cost_fct(thetas, X_scale, Y_scale, row_X, col_X, alpha):
    i = 0
    somme = 0
    while (i < row_X):
        somme += squar(hypo(X_sclale, i, thetas, col_X), Y_scale[i][0])
        i += 1
    return ()






















def make_predi(thetas, tmp_thetas, X_scale, Y_scale, row_X, col_X):
    alpha = 0.01
    

def main():
    data, X, Y, thetas, tmp_thetas = read_file()
    X_scale = np.array(X, dtype=float)
    Y_scale = np.array(Y, dtype=float)
    row_X = X.shape[0]
    col_X = X.shape[1]
    j = 1
    while (j < col_X):
        min_X = get_min_max(X, j, row_X, 1)
        max_X = get_min_max(X, j, row_X, 0)
        scale_feat(X_scale, j, row_X, min_X, max_X)
        j += 1
    min_Y = get_min_max(Y, 0, row_X, 1)
    max_Y = get_min_max(Y, 0, row_X, 0)
    scale_feat(Y_scale, 0, row_X, min_Y, max_Y)
    make_predi(thetas, tmp_thetas, X_scale, Y_scale, row_X, col_X)
  #  print(min_Y, max_Y, Y_scale)



if __name__ == "__main__":
    main()
