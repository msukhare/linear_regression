import argparse
import sys
import os
import pandas as pd
import numpy as np

from easyML import LinearReg, scaling_features, split_data

def read_data_csv(path_to_data, Y_name, seed=234):
    data = pd.read_csv(path_to_data)
    if Y_name not in data.keys():
        raise Exception('%s not present in %s' %(Y_name, path_to_data))
    if data.empty is True:
        raise Exception('%s is empty' %path_to_data)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    return np.asarray(data.drop(Y_name, axis=1).reset_index(drop=True)), np.asarray(data[Y_name])

def main(args):
    try:
        X, Y = read_data_csv(args.data_path, args.Y_name)
    except Exception as error:
        sys.exit('Error: ' + str(error))
    X, params_to_save = scaling_features(X, None, args.type_of_features_scaling)
    if args.evaluate is True:
        X_train, X_test, Y_train, Y_test = split_data(X, Y)
    else:
        X_train, Y_train = X, Y
    regressor = LinearReg(args.optimizer,\
                        args.learning_rate,\
                        args.cost_function,\
                        args.epochs,\
                        args.early_stopping,\
                        args.validation_fraction,\
                        args.n_epochs_no_change,\
                        args.tol,\
                        args.show_training)
    try:
        regressor.fit(X_train, Y_train)
    except Exception as error:
        sys.exit('Error:' + str(error))
    if args.evaluate is True:
        regressor.eval(X_test, Y_test)
    regressor.save_weights(args.file_where_store_weights, params_to_save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',\
            nargs='?',\
            type=str,\
            help="""correspond to path of csv file""")
    parser.add_argument('Y_name',\
            nargs='?',\
            type=str,\
            help="""correspond to name of column to predict in data""")
    parser.add_argument('--file_where_store_weights',\
            nargs='?',\
            type=str,\
            help="""correspond to path where store weights after training and
                informations about pipeline""")
    parser.add_argument('--optimizer',\
            nargs='?',\
            type=str,\
            default='gradient_descent',\
            choices=['gradient_descent', 'normal_equation'],\
            help="""correspond to back end to use during training""")
    parser.add_argument('--cost_function',\
            nargs='?',\
            type=str,\
            default='MSE',\
            choices=['MSE', 'RMSE', 'MAE'],\
            help="""correspond to cost function to use during training gradient descent
                and evaluation. By default MSE""")
    parser.add_argument('--learning_rate',\
            nargs='?',\
            type=float,\
            default=0.1,\
            help="""correspond to learning rate used during training if using gradient descent.
                By default 0.1""")
    parser.add_argument('--epochs',\
            nargs='?',\
            type=int,\
            default=100,\
            help="""correspond to numbers of epochs to do during training if using gradient descent.
                By default 100""")
    parser.add_argument('--early_stopping',\
            dest='early_stopping',\
            action='store_true',
            help="""if pass as params will do early stopping on val set, base on tol and
                n_epochs_no_change in gradient descent""")
    parser.add_argument('--validation_fraction',\
            nargs='?',\
            type=float,\
            default=0.10,\
            help="""correspond to percentage data use during training as val set in gradient descent.
                Used if early_stopping is True. By default 0.10 percentage of data""")
    parser.add_argument('--n_epochs_no_change',\
            nargs='?',\
            type=int,\
            default=5,\
            help="""correspond to numbers of epochs wait until cost function don't change.
                Only used in gradient descent and if --early_stoping is set at True.
                By default 5 epochs""")
    parser.add_argument('--tol',\
            nargs='?',\
            type=float,\
            default=1e-3,\
            help="""correspond to stopping criteron in early stopping.
                Only used in gradient descent and if --early_stopping is set at True.
                By default 1e-3""")
    parser.add_argument('--type_of_features_scaling',\
            nargs='?',\
            type=str,\
            default="standardization",\
            choices=['standardization', 'rescaling', 'normalization'],\
            help="""correspond to technic use for features scaling. By default standardization""")
    parser.add_argument('--show_training',\
            dest='show_training',\
            action='store_true',
            help="""if pass as params will show evolution of cost function during epochs 
                if using gradient descent as back-end""")
    parser.add_argument('--evaluate',\
            dest='evaluate',\
            action='store_true',
            help="""if pass as params will show scores obtains after training on 20 percent of
                data""")
    parsed_args = parser.parse_args()
    if parsed_args.data_path is None:
        sys.exit("Error: missing name of CSV data to use")
    if os.path.exists(parsed_args.data_path) is False:
        sys.exit("Error: %s doesn't exists" %parsed_args.data_path)
    if parsed_args.Y_name is None:
        sys.exit("Error: missing name of column to predict")
    main(parsed_args)
