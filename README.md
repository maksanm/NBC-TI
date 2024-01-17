# TI-NBC clustering

Implementation and analysis of **Neighborhood-Based Clustering by Means of the Triangle Inequality** comparing to standard **NBC** and **NBC** provided by *sklearn*.

## Requirements

- clustering-benchmarks
- scikit-learn
- scipy
- matplotlib
- numpy

## Project and models usage

Jupyter notebooks contain algorithm tests for different datasets and `k` parameter values.

The implemented models can be found in the `src/models` directory. Each model constructor accepts a `k` parameter value. The `fit_predict()` function of the models takes a dataset to be clustered and returns an array of labels. Examples of how to use these models with benchmark are demonstrated in the notebooks.

Precomputed results of reference algorithms for each *clustbench* dataset are stored in the `data/ref-precomputed-results` directory. I've selected the results of **KMeans** from *scikit-learn* for the comparison. You can find more [here](https://github.com/gagolews/clustering-results-v1) for download.


## Testing models with different datasets

[The *clustbench* dataset catalogue](https://clustering-benchmarks.gagolewski.com/weave/data-v1.html#wut/circles) contains a variety of 2D and 3D datasets, offering extensive possibilities for testing clustering algorithms. To use a different dataset, modify the `battery` and `dataset` variables in any of the notebooks.
