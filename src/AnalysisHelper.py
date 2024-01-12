import genieclust
from genieclust.compare_partitions import confusion_matrix, compare_partitions
import matplotlib.pyplot as plt
from tabulate import tabulate


def plot_results(methods, data, results_dict, reference=None, fig_size = 6):
    nrows = max(len(results_dict[method]) for method in results_dict)
    ncols = len(methods) + (int)(reference is not None)
    fig = plt.figure(figsize=(ncols * fig_size, nrows * fig_size))
    gs = fig.add_gridspec(nrows, ncols)
    for m, method in enumerate(methods):
        for i, k in enumerate(results_dict[method].keys()):
            fig.add_subplot(gs[i, m])
            genieclust.plots.plot_scatter(
                data, labels=results_dict[method][k], title=f"{method}; k={k}", axis="equal")
    if reference is not None:
        fig.add_subplot(gs[0, ncols - 1])
        genieclust.plots.plot_scatter(data, labels=reference, title="Reference partition assigned by experts", axis="equal")
    plt.show()

def confusion_matricies_table(methods, results_dict, reference):
    max_results_num = max(len(results_dict[method]) for method in results_dict)
    matricies_table = [[] for _ in range(max_results_num)]
    for method in methods:
        for i, k in enumerate(results_dict[method].keys()):
            conf_matrix = confusion_matrix(reference, results_dict[method][k])
            matricies_table[i].append("{} k={}:\n{}".format(method, k, conf_matrix))
    print(tabulate(matricies_table, tablefmt='fancy_grid'))

def measures(methods, results_dict, reference):
    max_results_num = max(len(results_dict[method]) for method in results_dict)
    statistics_table = [[] for _ in range(max_results_num)]
    from tabulate import tabulate
    for method in methods:
        for i, k in enumerate(results_dict[method].keys()):
            measures = compare_partitions(results_dict[method][k], reference)
            measures = '\n'.join(f'{k}: {v}' for k, v in measures.items())
            statistics_table[i].append("{} k={}:\n{}".format(method, k, measures))
    print(tabulate(statistics_table, tablefmt='fancy_grid'))