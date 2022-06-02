import math
import random
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy
import pandas
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from ..helpers._io import read_and_resize
from ..helpers._visualize import random_tile_collage
from ._visualize import FIGURE_HEIGHT, FIGURE_WIDTH

__all__ = ["OutlierDetector"]


class OutlierDetector:
    def __init__(
        self,
        metadata: pandas.DataFrame,
        num_clusters: int = 20,
        batch_size: int = 2**14,
        **kwargs,
    ):
        """Automatic outlier detection from color channel quantile metrics.

        Kmeans++ is used to cluster each tile into the desired number of
        clusters. Clusters are ordered by the distance to the origo in
        decreasing order, and thus the most likely outlier group is first and
        last groups contain normal images.

        Args:
            metadata: Metadata containing columns: `[channel_name]_q={quantile}`
            num_clusters: Number of clusters. Defaults to 20.
            batch_size: Batch size for kmeans++. Defaults to 2**14.

        Example:
            ```python
            import histoprep

            detector = histoprep.OutlierDetector(metadata, num_clusters=20)
            detect.plot_clusters(min_distance=10)
            # From the plots we might see that clusters 0-3 contain outliers.
            metadata["outlier"] = False
            metadata.loc[detector.clusters < 4, "outlier"] = True
            ```
        """
        self.__num_clusters = num_clusters
        # Pull paths from metadata.
        if "path" not in metadata.columns:
            raise ValueError("Metadata should not contain a 'path' column.")
        self.__all_paths = metadata["path"].to_numpy()
        # Then quantile metrics.
        X = metadata[[x for x in metadata.columns if "_q=" in x]].to_numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        self.__metrics = X
        # Cluster.
        self.__clusters = self.__cluster_data(
            X,
            n_clusters=num_clusters,
            batch_size=batch_size,
            **kwargs,
        )

    @property
    def metrics(self) -> numpy.ndarray:
        """Quantile metrics normalised with: ``(X - X.mean(0)) / X.std(0)``"""
        return self.__metrics

    @property
    def num_clusters(self) -> int:
        return self.__num_clusters

    @property
    def clusters(self) -> numpy.ndarray:
        """Cluster assignments."""
        return self.__clusters["clusters"]

    @property
    def cluster_counts(self) -> numpy.ndarray:
        """Number of members in each cluster."""
        return self.__clusters["counts"]

    @property
    def cluster_distances(self) -> numpy.ndarray:
        """Cluster mean euclidean distance to the origo."""
        return self.__clusters["distances"]

    @staticmethod
    def __cluster_data(X: numpy.ndarray, **kwargs) -> Dict[str, numpy.ndarray]:
        """Helper function to cluster metrics."""
        # Define cluster instance.
        kmeans = MiniBatchKMeans(**kwargs)
        # Cluster, get counts and calculate distance from origo.
        clusters = kmeans.fit_predict(X)
        __, counts = numpy.unique(clusters, return_counts=True)
        distances = numpy.array([numpy.linalg.norm(x) for x in kmeans.cluster_centers_])
        # Rename starting from the most distant.
        new_clusters = numpy.zeros_like(clusters)
        for new_idx, old_idx in enumerate(distances.argsort()[::-1]):
            new_clusters[clusters == old_idx] = new_idx
        # Cache clusters.
        return {
            "clusters": new_clusters,
            "counts": counts[distances.argsort()[::-1]],
            "distances": distances[distances.argsort()[::-1]],
        }

    def plot_clusters(
        self,
        min_distance: int = None,
        num_examples: int = 256,
        ncols: int = 32,
        px: int = 64,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 1.5, FIGURE_HEIGHT),
    ):
        """Plot example images from clusters.

        Args:
            min_distance: Minimum distance to filter which groups to plot.
                Defaults to None.
            num_examples: Number of example images from each group. Defaults to 256.
            ncols: Number of columns for plotting example images. Defaults to 32.
            px: Size of each example image. Defaults to 64.
            figsize: Figure size for plotting. Defaults to
                (default_width * 1.5, default_height)
        """
        # Collect all paths.
        paths = []
        for idx, dist in enumerate(self.cluster_distances):
            if min_distance is not None and dist < min_distance:
                break
            # Load cluster paths and select num_examples.
            cluster_paths = self.__all_paths[self.clusters == idx]
            if len(cluster_paths) < num_examples:
                cluster_paths = cluster_paths.tolist()
                cluster_paths += [None] * (num_examples - len(cluster_paths))
            else:
                cluster_paths = numpy.random.choice(
                    cluster_paths, size=num_examples
                ).tolist()
            paths += cluster_paths
        # Start reading images and plot as they are loaded.
        row = []
        collage = []
        cluster_idx = 0
        for tile in read_and_resize(paths, px):
            row.append(tile)
            if len(row) == ncols:
                collage.append(numpy.hstack(row))
                row = []
            if len(collage) == math.ceil(num_examples / 32):
                # Create collage and plot.
                collage = Image.fromarray(numpy.vstack(collage))
                plt.figure(figsize=figsize)
                plt.imshow(collage)
                plt.title(
                    "Cluster {} (images={} - distance={:.3f})".format(
                        cluster_idx,
                        self.cluster_counts[cluster_idx],
                        self.cluster_distances[cluster_idx],
                    )
                )
                plt.axis("off")
                plt.tight_layout()
                plt.show()
                # Reset collage.
                collage = []
                cluster_idx += 1

    def plot_cluster(
        self,
        cluster: int = 0,
        num_examples: int = 256,
        ncols: int = 32,
        px: int = 64,
        figsize: Tuple[float, float] = (FIGURE_WIDTH * 1.5, FIGURE_HEIGHT),
    ):
        """Plot example images from a single cluster.

        Args:
            cluster: Cluster number. Defaults to 0.
            num_examples: Number of example images from each group. Defaults to 256.
            ncols: Number of columns for plotting example images. Defaults to 32.
            px: Size of each example image. Defaults to 64.
            figsize: Figure size for plotting. Defaults to
                (default_width * 1.5, default_height)
        """
        # Collect paths.
        cluster_paths = self.__all_paths[self.clusters == cluster]
        if len(cluster_paths) == 0:
            warnings.warn("No image paths found for cluster {}.".format(cluster))
            return
        # Plot.
        plt.figure(figsize=figsize)
        plt.imshow(
            random_tile_collage(
                paths=cluster_paths,
                nrows=num_examples // ncols + 1,
                ncols=ncols,
                px=px,
            )
        )
        plt.title(
            "Cluster {} (images={} - distance={:.3f})".format(
                cluster,
                self.cluster_counts[cluster],
                self.cluster_distances[cluster],
            )
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def umap_representation(
        self,
        num_components: int = 2,
        num_neighbors: int = 15,
        metric: str = "euclidean",
        max_samples: int = 100_000,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Run Uniform Manifold Approximation and Projection (UMAP)
        dimensionality reduction.

        Args:
            num_components: Number of components, two offers easy visualisation.
                Defaults to 2.
            num_neighbors: Number of neighbours, should be between 2-100.
                Defaults to 15.
            max_samples: UMAP does not handle very large sample sizes well.
                Increase if you want to go and get coffee. Defaults to 100_000.
            **kwargs: Any keyword arguments for `umap.UMAP`.

        Returns:
            UMAP representation and indices of the selected samples
        """
        try:
            import umap
        except ImportError:
            warnings.warn(
                "Could not import `umap`. Please run `pip install umap-learn` "
                "to use this function."
            )
            return
        # Initialize.
        reducer = umap.UMAP(
            n_components=num_components,
            n_neighbors=num_neighbors,
            metric=metric,
            verbose=verbose,
            **kwargs,
        )
        # Fit and transform.
        if self.__metrics.shape[0] > max_samples:
            indices = numpy.array(
                random.choices(range(self.__metrics.shape[0]), k=max_samples)
            )
        else:
            indices = numpy.arange(self.__metrics.shape[0])
        representation = reducer.fit_transform(self.__metrics[indices, :])
        return representation, indices

    def pca_representation(
        self,
        num_components: int = 2,
        max_samples: int = 1_000_000,
        **kwargs,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Run princical component analysis (PCA) to get a representation of
        metrics. Does not produce as nice results as UMAP, but it's faster and
        supported out-of-the-box.

        Args:
            num_components: Number of components, two offers easy visualisation.
                Defaults to 2.
            max_samples: PCA does handle very large sample sizes well, but this argument
                can be used to limit the number of samples. Defaults to 1_000_000.
            **kwargs: Any keyword arguments for `sklearn.decomposition.PCA`.

        Returns:
            PCA representation and indices of the selected samples
        """
        # Initialize.
        reducer = PCA(n_components=num_components, **kwargs)
        # Fit and transform.
        if self.__metrics.shape[0] > max_samples:
            indices = numpy.array(
                random.choices(range(self.__metrics.shape[0]), k=max_samples)
            )
        else:
            indices = numpy.arange(self.__metrics.shape[0])
        representation = reducer.fit_transform(self.__metrics[indices, :])
        return representation, indices

    def plot_representation(
        self,
        coordinates: numpy.ndarray,
        indices: numpy.ndarray,
        label_clusters: bool = False,
        size: float = 0.5,
        cmap: str = None,
        figsize: Tuple[float, float] = (
            FIGURE_WIDTH * 1.5,
            FIGURE_HEIGHT * 1.5,
        ),
    ):
        """Plot representation of metrics.

        Args:
            coordinates: Representation coordinates.
            indices: Sample indices returned by `{umap, pca}_representation`.
            label_clusers: Label each cluster in the plot instead of log(distance).
            size: Size of each dot. Defaults to 0.5.
            cmap: Colour map. Defaults to "plasma" for distance and "coolwarm_r" for
                clusters.
            figsize: Figure size. Defaults to  Defaults to
                (default_width * 1.5, default_height * 1.5).
        """
        if label_clusters:
            labels = self.clusters[indices]
            cmap = "coolwarm_r" if cmap is None else cmap
        else:
            labels = numpy.log(numpy.linalg.norm(self.metrics[indices], axis=1))
            cmap = "plasma" if cmap is None else cmap
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        scatter = plt.scatter(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            s=size,
            cmap=cmap,
            c=labels,
            label=labels if label_clusters else None,
        )
        if label_clusters:
            ax.legend(
                *scatter.legend_elements(num=self.num_clusters - 1),
                title="Clusters",
                ncol=2,
            )
        else:
            plt.colorbar(label="log(distance)", shrink=0.75)
        ax.set_title("Representation of all tile metrics.")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        repr_str = "{}(num_clusters={}):\n".format(
            self.__class__.__name__, self.num_clusters
        )
        for idx, (count, dist) in enumerate(
            zip(self.cluster_counts, self.cluster_distances)
        ):
            repr_str += "{:>4}:  dist={:<7.2f} images={}\n".format(idx, dist, count)
            if idx > 9:
                repr_str += "{:>10}".format("...")
                break
        return repr_str
