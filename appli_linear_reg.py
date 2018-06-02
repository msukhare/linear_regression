import sys
import pandas as pd
import numpy as np

def read_file():
    try:
        file_content = pd.read_csv(sys.argv[1], header=None)
    except:
        sys.exit("file doesn't exist")
    nb_thetas = file_content.shape[0]
    tmp = file_content.iloc[0 : nb_thetas, :]
    to_ret = np.array(tmp.values, dtype=float)
    return (to_ret, nb_thetas)

def main():
    thetas, nb_thetas = read_file()
    if ((len(sys.argv) - 1) < nb_thetas):
         sys.exit("need more features")
    result = thetas[0][0]
    i = 1
    while (i < nb_thetas):
         result += thetas[i][0] * float(sys.argv[(i + 1)])
         i += 1
    print(result)

if __name__ == "__main__":
	main()
