import genieclust
import matplotlib.pyplot as plt

def plot_results(methods, data, results_dict, reference=None, fig_size = 6):
    nrows = max(len(results_dict[method]) for method in results_dict)
    ncols = len(methods) + (int)(reference is not None)

    fig = plt.figure(figsize=(ncols * fig_size, nrows * fig_size))
    gs = fig.add_gridspec(nrows, ncols)
    for m, method in enumerate(methods):
        for k, key in enumerate(results_dict[method].keys()):
            fig.add_subplot(gs[k, m])
            genieclust.plots.plot_scatter(
                data, labels=results_dict[method][key], title=f"{method}; k={key}", axis="equal")
    if reference is not None:
        fig.add_subplot(gs[0, ncols - 1])
        genieclust.plots.plot_scatter(data, labels=reference, title="Reference partition", axis="equal")
    plt.show()