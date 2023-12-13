"""
Make visuals (histograms of genre, theme, franchise)
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

"""
update_demographic_dict
Apply cumulative changes of frequency counts into dict as iteration occurs
Arguments: current (str) - instance of demographic being added
        dict0 (dict) - dictionary being modified
Returns:
        dict0
"""
def update_demographic_dict(current, dict0):
    if current in dict0:
        dict0[current] += 1
    else:
        dict0[current] = 1
    return dict0

"""
create_histogram
Create demographic (genre, theme, etc.) histogram based on classification (0 or 1)
Arguments: demographic (str)
        dict0, dict1 (dict) - dictionaries to combine for binary classification frequency visualization
        topN (int) - how many elements should be in the histogram
Returns: none (displays graph)
"""
def create_histogram(demographic, dict0, dict1, topN):
    tot = {**dict0, **dict1}

    # https://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary
    top5 = dict(Counter(tot).most_common(topN))  

    g0 = [k for k in top5 if k in dict0]
    g1 = [k for k in top5 if k in dict1]
    common_g = [k for k in g0 if k in g1]

    while len(common_g) < 4:
        topN += 1
        top_new = dict(Counter(tot).most_common(topN))  
        g0 = [k for k in top_new if k in dict0]
        g1 = [k for k in top_new if k in dict1]
        common_g = [k for k in g0 if k in g1]

    # remove "unknown" values
    unknown_key = "unknown"
    if unknown_key in common_g:
        common_g = [k for k in common_g if k != "unknown"]

    d0 = {}
    for i in common_g:
        d0[i] = dict0[i]
    d1 = {}
    for i in common_g:
        d1[i] = dict1[i]

    print(d0)
    print(d1)

    # https://www.geeksforgeeks.org/plotting-multiple-bar-charts-using-matplotlib-in-python/
    plotX = common_g
    x_axis = np.arange(len(plotX))
    plt.bar(x_axis - 0.2, d0.values(), 0.4, label='0')
    plt.bar(x_axis + 0.2, d1.values(), 0.4, label='1')
    plt.xticks(x_axis, plotX) 
    plt.xlabel(demographic) 
    plt.ylabel("Frequency")

    title_str = "Histogram of" + " " + demographic 
    plt.title(title_str) 
    plt.legend() 
    plt.show() 