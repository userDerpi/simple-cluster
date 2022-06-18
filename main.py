import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

if __name__ == '__main__':
    data = pd.read_csv('happyscore_income.csv')
    data.sort_values(by=['GDP'], inplace=True)

    richest = data[data.GDP > 1.2]  # pandas filter
    satisfaction_gdp = np.column_stack((richest.adjusted_satisfaction, richest.GDP))
    km_res = KMeans(n_clusters=3).fit(satisfaction_gdp)
    std_deviation = np.std(satisfaction_gdp, 0)  # calcs the standard deviation
    ellipses = [patches.Ellipse(cluster, *std_deviation * 2, alpha=0.25) for cluster in km_res.cluster_centers_]

    fig, graph = plt.subplots()  # to use ellipses in plot
    plt.xlabel('Adjusted satisfaction')
    plt.ylabel('GDP')
    graph.scatter(richest.adjusted_satisfaction, richest.GDP)  # .scatter(x,y) prepare the plot object

    # iterrows used to iterate a pandas dataframe
    for index, row in richest.iterrows():
        graph.text(row.adjusted_satisfaction,
                   row.GDP,
                   row.country)
    # .text(x,y,text) to add label to the points

    plt.scatter(km_res.cluster_centers_[:, 0], km_res.cluster_centers_[:, 1])

    for ellipse in ellipses:
        graph.add_patch(ellipse)

    plt.show()
