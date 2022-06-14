#!/usr/bin/env python3
"""lab2 template"""
from common import test_env as test_env
from common import feature_selection as feat_sel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


def print_graph(y_test, y_pred, title):
    """
    Dots close to the centre line ==> Better accurate
    """
    y_values = np.concatenate([y_test.flatten(), y_pred.flatten()])
    y_min, y_max, y_range = np.amin(
        y_values), np.amax(y_values), np.ptp(y_values)
    plt.scatter(y_pred, y_test, color='blue', alpha=0.5)
    plt.plot([y_min - y_range * 0.01, y_max + y_range * 0.01],
             [y_min - y_range * 0.01, y_max + y_range * 0.01], color='black')
    plt.xlim(y_min - y_range * 0.01, y_max + y_range * 0.01)
    plt.ylim(y_min - y_range * 0.01, y_max + y_range * 0.01)
    plt.ylabel('Actual Values (Black line)')
    plt.xlabel('Predicted Values (Blue dots)')
    plt.title(title)
    plt.grid()
    plt.show()


def print_metrics(y_true, y_pred, label):
    # Feel free to extend it with additional metrics from sklearn.metrics
    print('%s R^2: %.2f' % (label, r2_score(y_true, y_pred)))
    '''
    R^2 = 1 - (y - y_pred)^2 / (y - mean(y))^2
    Express fitting rate in percentage ( close to 1 is better - 100%)
    '''
    print('%s MSE: %.2f' % (label, mean_squared_error(y_true, y_pred)))
    print('%s RMSE: %.2f' % (label, np.sqrt(mean_squared_error(y_true, y_pred))))
    '''
    MSE = mean( (y_pred - y)^2 )
    Sensitive about outlier, therefore huge error will be emphasized ( close to 0 is better ) 
    
    RMSE = sprt(MSE)
    Still have the same characteristic feature as MSE but understandable to human ( close to 0 is better ) 
    '''
    print('%s MAE: %.2f' % (label, mean_absolute_error(y_true, y_pred)))
    '''
    MAE =  mean( abs(y_pred - y) )
    MAE Understandable for human, because the unit is not changed. ( close to 0 is better ) 
    '''
    # print('%s MSLE %.2f' % (label, mean_squared_log_error(y_true, y_pred)))
    '''
    MSLE = mean( (log(1+y_pred) - log(1+y)^2 ) 
    MSLE is useful when the range of error is huge. 
    Because those  error value will be squeezed into log values'. ( close to 0 is better ) 
    '''
    print('')


def linear_regression(X, y, print_text='Linear regression all in'):
    # Split train test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)

    reg = LinearRegression()
    reg.fit(X_train, y_train)  # Fit the train data
    y_pred = reg.predict(X_test)  # Predict with the test data
    print_metrics(y_test, y_pred, print_text)
    print_graph(y_test, y_pred, print_text)
    return reg


def linear_regression_selection(X, y):
    X_sel = feat_sel.backward_elimination(X, y, verbose=False)
    return linear_regression(X_sel, y, print_text='Linear regression with feature selection')


def polynomial_regression(X, y, dim=2):
    poly = PolynomialFeatures(degree=dim)
    X_pol = poly.fit_transform(X)
    return linear_regression(X_pol, y, print_text='Linear regression with polynomial: dimension(' + str(dim) + ')')


def svr_regressor(X, y, print_text='SVR regressor with rbf kernel'):
    sc = StandardScaler()
    X_standard = sc.fit_transform(X)  # Standard scaled data
    y_standard = sc.fit_transform(np.expand_dims(y, axis=1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_standard, y_standard, test_size=0.25, random_state=1)

    svr_rbf = SVR(kernel='rbf', gamma='auto')
    svr_rbf.fit(X_train, y_train)  # Train the train data
    y_pred = svr_rbf.predict(X_test)  # Predict with the test data
    print_metrics(np.squeeze(y_test), np.squeeze(y_pred), print_text)
    print_graph(np.squeeze(y_test), np.squeeze(y_pred), print_text)
    return svr_rbf


def decison_tree_regressor(X, y, print_test='decision tree'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print_metrics(y_test, y_pred, print_test)
    print_graph(y_test, y_pred, print_test)
    return dt


def random_forest(X, y, estimator, print_text='radom_forest'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)

    rf = RandomForestRegressor(n_estimators=estimator)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print_metrics(y_test, y_pred, print_text)
    print_graph(y_test, y_pred, print_text)
    return rf


if __name__ == '__main__':
    # Show environment info
    test_env.versions(
        ['numpy', 'statsmodels', 'sklearn', 'matplotlib', 'pandas'])

    # Load boston data and show some samples
    # https://scikit-learn.org/stable/datasets/index.html#boston-house-prices-dataset
    boston = load_boston()
    print(pd.DataFrame(boston.get('data'),
          columns=boston.get('feature_names')).sample(6))
    X = boston.get('data')
    y = boston.get('target')

    linear_regression(X, y)  # Linear regression
    # Linear regression with feature selection
    linear_regression_selection(X, y)
    polynomial_regression(X, y, dim=2)  # Polynomial regression
    svr_regressor(X, y)  # SVR
    decison_tree_regressor(X, y)  # Decision Tree
    random_forest(X, y, estimator=10)  # Random forest
    '''
    Notes:
    I tried Min Max scaling as well, although it is not effective in this case, 
    because dispersion goes wider because of the outlier.
    
    Considering the comparison of these scores, SVR with scaling is the best for predicting values.
    even though Random forest has slightly better R2 score.
    Because MAE and RMSE have smaller score, which indicate that they handle outlier in better way.
    If the feature scaling is applied to the random forest, it might have better result as well.
    '''

    print('Done')
