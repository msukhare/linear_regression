import argparse
import sys
import os
import pandas as pd
import numpy as np

from easyML import LinearReg, scaling_features

def main(args):
    try:
        regressor = LinearReg()
        pipeline = regressor.load_weights(parsed_args.path_to_weights)
        data, pipeline = scaling_features(np.asarray([[args.km]]), pipeline)
        hypo_price = regressor.predict(data)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    print("price of car with %f km is %f"%(args.km, hypo_price))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_weights',\
        nargs='?',\
        type=str,\
        help="""correspond to path with pipeline must be a .pth extension, must contain
                a pickle dict with info_about_pipeline, and weights keys""")
    parser.add_argument('km',\
        nargs='?',\
        type=float,\
        help="""correspond to data to use for predict price""")
    parsed_args = parser.parse_args()
    if parsed_args.path_to_weights is None:
        sys.exit("Error: missing path_to_weights")
    if os.path.exists(parsed_args.path_to_weights) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.path_to_weights)
    main(parsed_args)
