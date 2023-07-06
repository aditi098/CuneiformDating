import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


def plotPeriodWiseGraph():

    with open("../../../full_data/all_ids.json", 'r') as f:
        all_ids = json.load(f)

    df = pd.read_csv('../../../full_data/expanded_catalogue.csv')  

    periods = list(df['period.period'].unique())
    periods.remove(np.nan)
    periodWiseDict = dict()
    for period in periods:
        periodWiseDict[period] = []
    for index, row in df.iterrows():
        if row['isImagePresent'] == 1 and row['period.period'] in periods and str(row['id']) in all_ids:
            periodWiseDict[row['period.period']].append(row['id'])

    no_of_images = []
    periods = []
    for period in periodWiseDict.keys():
        no_of_images.append(len(periodWiseDict[period]))
        periods.append(period)

    Z = [(y,x) for y,x in sorted(zip(no_of_images,periods), reverse=True)]

    no_of_images = []
    periods = []

    for idx, (y,x) in enumerate(Z):
        no_of_images.append(y)
        periods.append(x)

    del df


    freq_series = pd.Series(no_of_images)

    # Plot the figure.
    plt.figure(figsize=(8, 8))
    ax = freq_series.plot(kind="bar")

    ax.set_xlabel("Periods")
    ax.set_ylabel("Number of images")
    ax.set_xticklabels(periods)


    rects = ax.patches

    # Make some labels.
    for rect, label in zip(rects, no_of_images):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 200, label, ha="center", va="bottom", rotation ="vertical"
        )

    plt.show()
    # plt.savefig('cuneiform.png')
