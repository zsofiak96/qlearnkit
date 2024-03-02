import numpy as np
from sklearn.exceptions import NotFittedError

from qlearnkit.algorithms import QKMeans


class QFCMeans(QKMeans):
    """Quantum-modeled fuzzy c-means algorithm for clustering."""

    @staticmethod
    def _next_centers(X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Update cluster centers."""
        um = u**2
        return (X.T @ um / np.sum(um, axis=0)).T

    def soft_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate membership values to each cluster center.

        Args:
                X: ndarray, data to predict

        Returns:
             np.ndarray, fuzzy partition array (n_samples rows and n_clusters columns)
        """
        circuits = self._construct_circuits(X)
        results = self.execute(circuits)
        temp = self._get_distances_centroids(results) ** (2 / (2 - 1))
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        np.nan_to_num(denominator_, nan=1)
        np.place(denominator_, denominator_ < 1, [1])
        denominator_ = temp[:, :, np.newaxis] / denominator_
        np.nan_to_num(denominator_, nan=1)
        np.place(denominator_, denominator_ < 1, [1])
        return 1 / denominator_.sum(2)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Fits the model using X as training dataset. The fit model creates clusters
        from the training dataset given as input.

        :param X: training dataset
        :return: trained QFCMeans object
        """
        self.X_train = np.asarray(X)
        # initialize membership values U
        self.rng = np.random.default_rng(self.random_state)
        n_samples = self.X_train.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(self.u.sum(axis=1)[np.newaxis].T, self.n_clusters)
        self.n_iter_ = 0
        error = np.inf

        while error > self.tol and self.n_iter_ < self.max_iter:
            u_old = self.u.copy()
            self.cluster_centers_ = self._next_centers(self.X_train, self.u)
            self.u = self.soft_predict(self.X_train)
            for i in range(self.u.shape[0]):
                self.u[i] = self.u[i] / self.u[i].sum()
            error = np.linalg.norm(self.u - u_old)
            self.n_iter_ = self.n_iter_ + 1
        self.labels_ = np.argmax(self.u, axis=1)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the labels of the provided data.

        Args:
                X: ndarray, test samples

        Returns:
            Index of the cluster each sample belongs to.
        """
        if self.labels_ is None:
            raise NotFittedError(
                "This QFCMeans instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator."
            )

        return self.soft_predict(X_test).argmax(axis=-1)
