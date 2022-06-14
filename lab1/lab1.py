#!/usr/bin/env python3
import sys

import pandas as pd
from matplotlib import pyplot as plt

from common import describe_data as describe_data
from common import test_env as test_env


def read_data(file):
    """Return pandas dataFrame read from csv file"""
    try:
        return pd.read_csv(file, sep=';', decimal=',', thousands=' ',
                           encoding='latin-1')
    except FileNotFoundError:
        sys.exit('ERROR: ' + file + ' not found')


if __name__ == '__main__':
    REQUIRED = ['matplotlib', 'pandas']
    test_env.versions(REQUIRED)
    df = read_data('data/tasutud_maksud_2021_ii_kvartal_eng.csv')

    # Print data overview with function print_overview in
    # common/describe_data.py
    describe_data.print_overview(df.sample(5))

    # Remove maximum printed rows limit. Otherwise next print is truncated
    pd.options.display.max_rows = None

    # Print all possible values with counts in column 'County' with help of
    # groupby() and size()
    print(df.groupby('County').size())

    # Print all unique values in column 'Type'
    print(df['Type'].unique())

    # Extract only SME (Small and medium-sized enterprises) data based on
    # number of employees
    sme_df = df[(df['Type'] == 'Company') & (
        df['Number of employees'] > 0) & (df['Number of employees'] < 250)]
    print(sme_df.sample(5))

    # Extract column 'Number of employees' to variable sme_employees
    sme_employees = sme_df['Number of employees']

    # Calculate mean to variable sme_employees_mean and print the value
    sme_employees_mean = sme_employees.mean()
    print(f'Mean: {sme_employees_mean}')

    # Calculate median to variable sme_employees_median and print the value
    sme_employees_median = sme_employees.median()
    print(f'Median: {sme_employees_median}')

    # Calculate mode to variable sme_employees_mode and print the value
    sme_employees_mode = sme_employees.mode()
    print(f'Mode: {sme_employees_mode}')

    # Calculate standard deviation to variable sme_employees_std and print the
    # value
    sme_employees_std = sme_employees.std()
    print(f'STD: {sme_employees_std}')

    # Calculate and print quartiles
    print(f'1 Q: {sme_employees.quantile(.25)}')
    print(f'2 Q: {sme_employees.quantile(.50)}')
    print(f'3 Q: {sme_employees.quantile(.75)}')

    # Draw dataset histogram including mean, median and mode
    figure_1 = 'SME Number of employees histogram'
    plt.figure(figure_1)
    plt.hist(
        sme_employees,
        range=(
            sme_employees.min(),
            sme_employees.max()),
        bins=50,
        edgecolor='black')
    plt.title(figure_1)
    plt.xlabel('SME Number of employees')
    plt.ylabel('Count')
    plt.axvline(sme_employees_mean, color='r', linestyle='solid',
                linewidth=1, label='Mean')
    plt.axvline(sme_employees_median, color='y', linestyle='dotted',
                linewidth=1, label='Median')
    plt.axvline(sme_employees_mode[0], color='orange',
                linestyle='dashed', linewidth=1, label='Mode')

    plt.legend()
    plt.savefig('results/figure_1.png')

    # Draw boxplot
    figure_2 = 'SME Number of employees box plot'
    plt.figure(figure_2)
    plt.boxplot(sme_employees)
    plt.ylabel('SME Number of employees')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.title(figure_2)
    plt.savefig('results/figure_2.png')

    # Find and print correlation matrix between 'Number of employees' and
    # 'Labour taxes and payments'
    print('Correlation')
    print(df[['Number of employees', 'Labour taxes and payments']].corr())

    # Plot correlation with scatter plot
    figure_3 = 'SME Number of employees and Labour taxes and payments correlation'
    plt.figure(figure_3)
    plt.suptitle(figure_3)
    plt.subplot(2, 2, 2)
    plt.scatter(sme_df['Number of employees'],
                sme_df['Labour taxes and payments'], color='red', s=0.5)
    plt.xlabel('Number of employees')
    plt.ylabel('Labour taxes and payments')
    plt.subplot(2, 2, 3)
    plt.scatter(
        sme_df['Labour taxes and payments'],
        sme_df['Number of employees'],
        color='red',
        s=0.5)
    plt.ylabel('Number of employees')
    plt.xlabel('Labour taxes and payments')
    plt.savefig('results/figure_3.png')

    # Show all figures in different windows
    plt.show()
    print('Done')
