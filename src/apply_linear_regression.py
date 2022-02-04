import argparse
import sys
import os
import pandas as pd
import numpy as np

from easyML import LinearReg, scaling_features

def is_float(to_check):
    return True

def main(args):
    regressor = LinearReg()
    try:
        if args.path_to_weights is not None:
            pipeline = regressor.load_weights(args.path_to_weights)
        else:
            pipeline = None
            regressor.init_weights(2)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    print("Enter mileage of car for a price estimation: ")
    for mileage in sys.stdin:
        if is_float(mileage) is True:
            data = np.asarray([[float(mileage)]])
            if pipeline is not None:
                data, pipeline = scaling_features(data, pipeline)
            hypo_price = regressor.predict(data)
            print("price of car with %s km is %f \n" %(mileage, hypo_price))
        else:
            print("input data must be a float or int \n")
        print("Enter new mileage of car for a new price estimation: ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_weights',\
        nargs='?',\
        type=str,\
        help="""correspond to path with pipeline must be a .pth extension, must contain
                a pickle dict with info_about_pipeline, and weights keys""")
    parsed_args = parser.parse_args()
    if parsed_args.path_to_weights is not None and\
        os.path.exists(parsed_args.path_to_weights) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.path_to_weights)
    if parsed_args.path_to_weights is not None and\
        os.path.isfile(parsed_args.path_to_weights) is False:
        sys.exit("Error: %s must be a file" %parsed_args.path_to_weights)
    main(parsed_args)
