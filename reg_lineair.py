# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    reg_lineair.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/05/30 13:44:17 by msukhare          #+#    #+#              #
#    Updated: 2018/05/30 17:06:15 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def read_file():
    fd = open("data.csv", "r")
    str_file = fd.read()
    to_ret = str_file.split('\n');
    del to_ret[-1]
    tmp = list(to_ret);
    del tmp[0]
    i = 0
    for ele in tmp:
        tmp1 = ele.split(',')
        j = 0
        for ele1 in tmp1:
            j += 1
        i += 1
    to_ret1 = np.eye(i, j)
    i = 0
    for ele in tmp:
        tmp1 = ele.split(',')
        j = 0
        for ele1 in tmp1:
            to_ret1[i][j] = ele1
            j += 1
        i += 1
    return (to_ret, to_ret1)

def main():
    graph, data = read_file()
    print(data)
if __name__ == "__main__":
    main()
