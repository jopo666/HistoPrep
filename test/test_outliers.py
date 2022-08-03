# import os

# import numpy
# import pandas
# import pytest
# from histoprep import OutlierDetector, OutlierVisualizer

# from paths import DATA_PATH

# metadata = pandas.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
# visual = OutlierVisualizer(metadata)
# detect = OutlierDetector(metadata)


# def test_visualiser():
#     visual.plot_rgb_std(4)
#     visual.plot_rgb_std(4, log_scale=True)
#     visual.plot_hsv_std(4)
#     visual.plot_hsv_std(4, log_scale=True)
#     visual.plot_rgb_quantiles(4)
#     visual.plot_rgb_quantiles(4, log_scale=True)
#     visual.plot_hsv_quantiles(4)
#     visual.plot_hsv_quantiles(4, log_scale=True)


# def test_detector():
#     assert len(detect.cluster_counts) == detect.num_clusters
#     assert len(detect.cluster_distances) == detect.num_clusters
#     assert isinstance(detect.metrics, numpy.ndarray)
#     assert all(numpy.isclose(detect.metrics.mean(0), 0))
#     assert len(detect.clusters) == detect.metrics.shape[0]
#     detect.plot_clusters(10)
#     detect.plot_cluster(0)
#     with pytest.warns(UserWarning):
#         detect.plot_cluster(100000)


# def test_umap_repr():
#     with pytest.warns():
#         detect.umap_representation(2, max_samples=100)


# def test_pca_repr():
#     coords, indices = detect.pca_representation(2, max_samples=100)
#     assert coords.shape == (100, 2)
#     assert indices.size == 100


# def test_plot_repr():
#     coords, indices = detect.pca_representation(2, max_samples=100)
#     detect.plot_representation(coords, indices)
