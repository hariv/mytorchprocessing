import pandas as pd
import pylab as plt
import sys

"""
Script that takes in a log file and a metric
such as loss or accuracy and and plots variation
of that metric with training iterations.
"""
def plot(log_file, metric):
    df = pd.DataFrame.from_csv(log_file)
    df.plot(x='iteration', y=['training_' + metric, 'val_' + metric])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python plot.py <log_file> <metric>')
        exit()
    plot(sys.argv[1], sys.argv[2])
