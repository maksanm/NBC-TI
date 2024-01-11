import genieclust
import matplotlib.pyplot as plt

def plot_results(methods, data, results_dict, reference=None, fig_size = 6):
    nrows = max(len(results_dict[method]) for method in results_dict)
    ncols = len(methods) + 1 * (reference is not None)
    fig, _ = plt.subplots(nrows, ncols, figsize=(ncols * fig_size, nrows * fig_size))
    for m, method in enumerate(methods):
        for k, key in enumerate(results_dict[method].keys()):
            fig.add_subplot(nrows, ncols, (m + 1)*(k + 1))
            genieclust.plots.plot_scatter(
                data, labels=results_dict[method][key], title=f"{method}; k={key}", axis="equal")
    if reference is not None:
        fig.add_subplot(nrows, ncols, ncols)
        genieclust.plots.plot_scatter(data, labels=reference, title="Reference partition", axis="equal")
    plt.show()