# NBC-TI clustering

Implementation and analysis of **Neighborhood-Based Clustering by Means of the Triangle Inequality** comparing to standard **NBC** and **NBC** provided by *sklearn*.

## Requirements

- clustbench
- scikit-learn
- scipy
- matplotlib
- numpy

## Project structure

Jupyter notebooks contain algorithm results for different datasets and `k` parameter values. Implemented models are located in `src/models`. Precomputed results of reference algorithms for each *clustbench* dataset are stored in `data/ref-precomputed-results`, results of **KMeans** from *sklearn* were selected for comparison, more can be found [here](https://github.com/gagolews/clustering-results-v1).

## Testing models with different datasets

[The *clustbench* dataset catalogue](https://clustering-benchmarks.gagolewski.com/weave/data-v1.html#wut/circles) contains a variety of 2D and 3D datasets, offering extensive possibilities for testing clustering algorithms. To use a different dataset, modify the `battery` and `dataset` variables in any of the notebooks.