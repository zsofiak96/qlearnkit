import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from qlearnkit.algorithms import QFCMeans
import pytest

from qiskit_aer import Aer
from qiskit.utils import QuantumInstance, algorithm_globals

seed = 42
algorithm_globals.random_seed = seed

sv_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator_statevector"),
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)

qasm_quantum_instance = QuantumInstance(
    Aer.get_backend("aer_simulator"),
    shots=100,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
)


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fcmeans_results(quantum_instance, quantum_instance_type, dtype):
    # Checks that FCMeans works as intended on toy dataset by comparing with
    # expected results computed by hand.
    X = np.array([
        [0, 0],
        [0.5, 0],
        [0.5, 1],
        [1, 1]
    ], dtype=dtype)

    init_centers = np.array([
        [0, 0],
        [1, 1]
    ], dtype=dtype)

    expected_labels = [0, 0, 1, 1]

    qfcmeans = QFCMeans(
        quantum_instance=quantum_instance,
        n_clusters=2,
        n_init=1,
        init=init_centers
    )
    qfcmeans.fit(X)

    np.testing.assert_array_equal(qfcmeans.labels_, expected_labels,
                                  f"Test failed with {quantum_instance_type}.\n"
                                  f"Expected {expected_labels}, but it was {qfcmeans.labels_}")


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
def test_fcmeans_relocated_clusters(quantum_instance, quantum_instance_type):
    # check that empty clusters are relocated as expected
    X = np.array([
        [0, 0],
        [0.5, 0],
        [0.5, 1],
        [1, 1]
    ])

    # second center too far from others points will be empty at first iter
    init_centers = np.array([
        [0.5, 0.5],
        [3, 3]
    ])

    expected_labels = [0, 0, 1, 1]

    qfcmeans = QFCMeans(
        quantum_instance=quantum_instance,
        n_clusters=2,
        n_init=1,
        init=init_centers
    )

    pred_labels = qfcmeans.fit_predict(X)

    np.testing.assert_array_equal(pred_labels, expected_labels,
                                  f"Test failed with {quantum_instance_type}.\n"
                                  f"Expected {expected_labels}, but it was {pred_labels}"
                                  )


@pytest.mark.parametrize(
    'quantum_instance, quantum_instance_type',
    [
        (qasm_quantum_instance, 'qasm'),
        (sv_quantum_instance, 'statevector')
    ]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fcmeans_iris(quantum_instance, quantum_instance_type, dtype):
    # test fcmeans using iris dataset (first 2 features)
    qfcmeans = QFCMeans(
        n_clusters=3,
        quantum_instance=quantum_instance
    )
    X, y = load_iris(return_X_y=True)
    X = np.asarray([x[0:2] for x, y_ in zip(X, y) if y_ != 2],
                   dtype=dtype)
    y = np.asarray([y_ for x, y_ in zip(X, y) if y_ != 2],
                   dtype=dtype)

    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.1, random_state=42)

    # Perform clustering
    qfcmeans.fit(train_data, train_label)

    # Predict the nearest cluster of train data
    predicted_labels = qfcmeans.predict(train_data)

    # assert fit(X).predict(X) equal fit_predict(X)
    np.testing.assert_array_equal(predicted_labels, qfcmeans.labels_)