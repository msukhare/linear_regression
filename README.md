# linear_regression

## About linear_regression.

* linear_regression is a machine learning project.

* linear_regression is composed of two scripts, `linear_reg.py` and `appli_linear_reg.py`.

### About `linear_reg.py`.

* `linear_reg.py` trains thetas to predict prize of house or cars.

* It writes thetas in a file after trained them.

* `linear_reg.py` use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to minimize cost function.

* I use mean square error as cost function.

### About `appli_linear_reg.py`.

* `appli_linear_reg.py' use trained thetas to predict a new prize acording to the new features passed in parameters.

* It writes prediction in the standard output.

### About data.csv you can use.

* `data.csv` has one feature and Y. Feature is mileage, and Y is prize of car.

* `ex1data2.csv` has two features and Y. First feature is size of house, second is number of rooms in house, and Y is prize of house.

* You can create your own data.csv but the format must be [x1, x2, x3 ..., xm, Y]. x is the features, and Y is what you want predict and must be the last column in csv.

## What do you need to make linear_regression work ?

* python => 3.0

* [numpy](http://www.numpy.org/)

* [pandas](https://pandas.pydata.org/)

## Usage:

### `linear_reg.py`

* `python3 linear_reg [Data.csv] [FileContainNewThetas]`. `FileContainNewThetas` is a file which contain trained thetas and used by `appli_linear_reg.py` to predict new Y.

### `aplli_linear_reg`

* `python3 appli_linear_reg [FileContainThetas] [Features ...]`. `FileContainThetas` is the file created by `linear_reg.py` after training. The number of features must be the same as the number of features in data.csv which used to train thetas.
