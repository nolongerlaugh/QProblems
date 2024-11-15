import pyideem
import pickle as pkl
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.qasm2 import dump
from pathlib import Path
import numpy as np

PATH_INPUT = Path('/workspace/qasm_circuit.qasm')
PATH_OUTPUT = Path('/workspace/test_qasm_circuit_output.json')
DATA_PATH = Path('/workspace/user_data_file.txt')

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')
Y_test = np.load('Y_test.npy')
num_qubits = 4


fmap = ZZFeatureMap(num_qubits, reps=2)
ansatz = RealAmplitudes(num_qubits, reps=3, entanglement='circular')
params = ParameterVector('θ', ansatz.num_parameters)
ansatz.assign_parameters(params, inplace=True)


def qiskit_circuit_constructor(weights, data_vector):
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    fmap_bound = fmap.assign_parameters({fmap.parameters[i]: data_vector[i] for i in range(len(data_vector))})
    ansatz_bound = ansatz.assign_parameters({params[i]: weights[i] for i in range(len(weights))})
    
    qc.compose(fmap_bound.compose(ansatz_bound), inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


losses_train = []
losses_test = []

def train_test_loss(weights, X_train, X_test, Y_train, Y_test):
    lossi = 0
    test_loss = 0
    for data_vector, label in zip(X_train, Y_train):
        dump(qiskit_circuit_constructor(weights, data_vector).decompose().decompose(), PATH_INPUT)
        
        qc = pyideem.QuantumCircuit.loadQASMFile(str(PATH_INPUT))
        backend = pyideem.StateVector(num_qubits)
        counts = qc.execute(1000, backend, noise_cfg=None, return_memory=False).counts
        
        bitstring = max(list(counts.items()), key=lambda x: x[1])[0]
        ones = bitstring.count('1')
        if ones >= num_qubits - ones:
            lossi += (label - 1)**2
        else:
            lossi += label**2
    
    for data_vector, label in zip(X_test, Y_test):
        dump(qiskit_circuit_constructor(weights, data_vector).decompose().decompose(), PATH_INPUT)
        
        qc = pyideem.QuantumCircuit.loadQASMFile(str(PATH_INPUT))
        backend = pyideem.StateVector(num_qubits)
        counts = qc.execute(1000, backend, noise_cfg=None, return_memory=False).counts
        
        bitstring = max(list(counts.items()), key=lambda x: x[1])[0]
        ones = bitstring.count('1')
        if ones >= num_qubits - ones:
            test_loss += (label - 1)**2
        else:
            test_loss += label**2
    losses_train.append(lossi / len(X_train))
    losses_test.append(test_loss / len(X_test))
    return lossi / len(X_train)


def optimize_parameters(initial_params, X_train, X_test, Y_train, Y_test):
    result = minimize(
        fun=train_test_loss,
        x0=initial_params,
        args=(X_train, X_test, Y_train, Y_test),
        method='COBYLA',
        options={'maxiter': 100, 'disp': True}
    )
    return result.x, result.fun

np.random.seed(52)
initial_weights = 2 * np.pi * np.random.uniform(size=ansatz.num_parameters)
optimal_params, optimal_loss = optimize_parameters(initial_weights, X_train, X_test, Y_train, Y_test)
print("Оптимальные параметры:", optimal_params)
print("Минимальная функция потерь:", optimal_loss)
np.save('losses_train.npy', losses_train)
np.save('losses_test.npy', losses_test)