import genieclust
from genieclust.compare_partitions import confusion_matrix, compare_partitions
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from tabulate import tabulate
import numpy as np

colors = list(TABLEAU_COLORS.keys())

def plot_results(data, methods, results_dict, reference=None, fig_size=6):
    n_clusters_ref = len(np.unique(reference))
    k_min = n_clusters_ref
    n = len(reference)
    nrows = max(len(results_dict[method]) for method in methods)
    ncols = len(methods) + (int)(reference is not None)

    fig = plt.figure(figsize=(ncols*fig_size, nrows*fig_size))
    gs = fig.add_gridspec(nrows, ncols)
    dim = data.shape[1]
    if dim == 2:
        for m, method in enumerate(methods):
            for i, k in enumerate(results_dict[method].keys()):
                fig.add_subplot(gs[i, m])
                n_clusters = len(np.unique(results_dict[method][k]))
                genieclust.plots.plot_scatter(
                    data, labels=results_dict[method][k], title=f"{method}; n={n}; n_clusters={n_clusters}", axis="equal")
                if reference is not None and k == n_clusters_ref:
                    fig.add_subplot(gs[i, ncols - 1])
                    genieclust.plots.plot_scatter(data, labels=reference,
                                                title=f"Reference partition assigned by experts; n_clusters={n_clusters_ref}", axis="equal")
    elif dim == 3:
        for m, method in enumerate(methods):
            for i, k in enumerate(results_dict[method].keys()):
                k_min = min(k, k_min)
                n_clusters = len(np.unique(results_dict[method][k]))
                ax = fig.add_subplot(gs[i, m], projection='3d', title=f"{method}; n={n}; n_clusters={n_clusters}")
                for it, p in enumerate(data):
                    ax.scatter(p[0], p[1], p[2], color=colors[results_dict[method][k][it] - 1])
        if reference is not None:
            ax = fig.add_subplot(gs[0, ncols - 1], projection='3d', title=f"Reference partition assigned by experts; n_clusters={n_clusters_ref}")
            for it, label in enumerate(reference):
                ax.scatter(data[it][n_clusters_ref - k_min], data[it][1], data[it][2], color=colors[label - 1])
    plt.show()

def confusion_matricies_table(methods, results_dict, reference, title=None):
    k_ref = len(np.unique(reference))
    matricies_table = [[]]
    for method in methods:
        conf_matrix = confusion_matrix(reference, results_dict[method][k_ref])
        matricies_table[0].append("{}\n{}".format(method, conf_matrix))
    print(f"{title}\n" + tabulate(matricies_table, tablefmt='fancy_grid'))

def measures(methods, results_dict, reference, title=None):
    k_ref = len(np.unique(reference))
    statistics_table = [[]]
    from tabulate import tabulate
    for method in methods:
        measures = compare_partitions(results_dict[method][k_ref], reference)
        measures = '\n'.join(f'{k}: {v}' for k, v in measures.items())
        statistics_table[0].append("{}\n{}".format(method, measures))
    print(f"{title}\n" + tabulate(statistics_table, tablefmt='fancy_grid'))