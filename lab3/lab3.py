#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd

# STUDENT SHALL ADD NEEDED IMPORTS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from common import describe_data, test_env
from common.classification_metrics import print_metrics


def read_data(file):
    """Return pandas dataFrame read from Excel file"""
    try:
        return pd.read_excel(file, header=0)
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


def preprocess_data(df, verbose=False):
    y_column = 'In university after 4 semesters'

    # Features can be excluded by adding column name to list
    drop_columns = []

    categorical_columns = [
        'Faculty',
        'Paid tuition',
        'Study load',
        'Previous school level',
        'Previous school study language',
        'Recognition',
        'Study language',
        'Foreign student'
    ]

    # Handle dependent variable
    if verbose:
        print('Missing y values: ', df[y_column].isna().sum())

    y = df[y_column].values
    # Encode y. Naive solution
    y = np.where(y == 'No', 0, y)
    y = np.where(y == 'Yes', 1, y)
    y = y.astype(float)

    # Drop also dependent variable variable column to leave only features
    drop_columns.append(y_column)
    df = df.drop(labels=drop_columns, axis=1)

    # Remove drop columns for categorical columns just in case
    categorical_columns = [
        i for i in categorical_columns if i not in drop_columns]

    # STUDENT SHALL ENCODE CATEGORICAL FEATURES
    categorical_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    df = df.drop(labels=categorical_columns, axis=1)
    df = df.join(categorical_df)

    # Process numerical columns
    numerical_columns = [
        'Estonian language exam points',
        'Estonian as second language exam points',
        'Mother tongue exam points',
        'Narrow mathematics exam points',
        'Wide mathematics exam points',
        'Mathematics exam points'
    ]
    # Summarize Numerical features
    lang_exams = df[numerical_columns[0:3]].values
    math_exams = df[numerical_columns[3:6]].values
    best_lang_exam = []
    best_math_exam = []
    for row in range(lang_exams.shape[0]):
        best_lang_exam.append(np.nanmax(lang_exams[row]))  # Choose max value
        best_math_exam.append(np.nanmax(math_exams[row]))  # Leave NaN as it is

    # Min-Max Scaling (Max 100, Min 0)
    best_lang_exam = [best / 100 for best in best_lang_exam]
    best_math_exam = [best / 100 for best in best_math_exam]

    # Update df
    df = df.drop(labels=numerical_columns, axis=1)  # Drop the old numerical
    df['Language_exam'] = best_lang_exam  # Replace with new columns
    df['Math_exam'] = best_math_exam

    # Handle missing data. At this point only exam points should be missing
    df[['Language_exam', 'Math_exam']] = df[[
        'Language_exam', 'Math_exam']].fillna(value='Missing')

    # It seems to be easier to fill whole data frame as only particular columns
    if verbose:
        describe_data.print_nan_counts(df)

    # STUDENT SHALL HANDLE MISSING VALUES
    df = df.replace('Missing', 0)

    if verbose:
        describe_data.print_nan_counts(df)

    # Return features data frame and dependent variable
    return df, y


# STUDENT SHALL CREATE FUNCTIONS FOR LOGISTIC REGRESSION CLASSIFIER, KNN
# CLASSIFIER, SVM CLASSIFIER, NAIVE BAYES CLASSIFIER, DECISION TREE
# CLASSIFIER AND RANDOM FOREST CLASSIFIER

def logistic_regression(X_train, y_train, X_test):
    """
    L1 regularization : Choose features to drop or not
    L2 regularization : Smooth every features velue to prevent over fitting
    I don't see any difference between results from changing solver in this case
    """
    clf = LogisticRegression(solver='saga', penalty='l2',
                             C=0.7).fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


def k_neighbors(X_train, y_train, X_test):
    """
    Increasing n_neighbors might lead to over fitting
    """
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


def svc(X_train, y_train, X_test):
    """
    too small gamma --> under fitting
    too big gamma --> over fitting
    """
    clf = SVC(kernel='sigmoid', gamma=0.7)
    clf.fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


def nb(X_train, y_train, X_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


def decision_tree(X_train, y_trian, X_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


def random_forest(X_train, y_train, X_test):
    """
    Increasing n_estimators make accuracy better but, make the code slower
    """
    clf = RandomForestClassifier(random_state=0, n_estimators=50)
    clf.fit(X_train, y_train)
    print(f'Training Score:{clf.score(X_train, y_train)}')
    return clf.predict(X_test)


if __name__ == '__main__':
    modules = ['numpy', 'pandas', 'sklearn']
    test_env.versions(modules)

    students = read_data('data/students.xlsx')
    # STUDENT SHALL CALL PRINT_OVERVIEW AND PRINT_CATEGORICAL FUNCTIONS WITH
    # FILE NAME AS ARGUMENT
    students_X, students_y = preprocess_data(students)
    describe_data.print_overview(
        students, file='results/students_overview.txt')
    describe_data.print_categorical(
        students, file='results/students_categorical_data.txt')

    # STUDENT SHALL CALL CREATED CLASSIFIERS FUNCTIONS
    X_train, X_test, y_train, y_test = train_test_split(students_X, students_y,
                                                        test_size=0.25, random_state=8)
    tmp = 0
    # Logistic regression
    y_pred = logistic_regression(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='LOGISTIC REGRESSION', verbose=tmp)

    # K neighbors
    y_pred = logistic_regression(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='K-Neighbors', verbose=tmp)

    # SVC
    y_pred = svc(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='SVC', verbose=tmp)

    # Naive Bayes
    y_pred = nb(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='Multinomial NB', verbose=tmp)

    # Decision Tree
    y_pred = decision_tree(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='Decision tree', verbose=tmp)

    # Random forest
    y_pred = random_forest(X_train, y_train, X_test)
    print_metrics(y_test, y_pred, label='Random forest', verbose=tmp)

    print('Done')
