### Description of the project
This project implements the optimization of routes for buses in the transport network using the quadratic unlimited binary optimization (QUBO) method. The code generates a QUBO matrix based on the specified constraints and the objective function, and then solves the optimization problem using the pyqiopt library.


### Files
`task-2-adjacency_matrix.csv`: A CSV file containing the adjacency matrix for the transport network.

`task-2-nodes.csv`: A CSV file containing a list of nodes (stops) and the number of people at each node.

## Quick start

### Step 1: Uploading and processing data

```python
import pandas as pd
import numpy as np
import pyqiopt as pq

# Загрузка данных
adj = pd.read_csv("/workspace/kirill/task-2-adjacency_matrix.csv")
nodes = pd.read_csv("/workspace/kirill/task-2-nodes.csv", names=["name", "n_people"])

# Предобработка данных
nodes.sort_values(by='n_people', inplace=True)
nodes = nodes.reset_index(drop=True)

name_to_index = {nodes['name'][i]: i for i in nodes.index}
index_to_name = {i: nodes['name'][i] for i in nodes.index}

people_count = nodes['n_people'].to_numpy()
N_LOCS = nodes.shape[0]
N_CROSSROADS = N_LOCS - np.count_nonzero(nodes['n_people'])
N_BUS = 15
N_TIMES = 16

# Обработка матрицы смежности
adj_ = adj.set_index("Unnamed: 0")
edges, no_edges = [], []
cost_matrix = np.zeros((N_LOCS, N_LOCS))
for i in range(N_LOCS):
    for j in range(i + 1, N_LOCS):
        char = adj_.iloc[i, j]
        if char != '-':
            edges.append((i, j, int(char)))
        else:
            no_edges.extend([(i, j), (j, i)])
```
### Step 2: Defining the task parameters

```python
N_ANCILLA_PER_BUS = 4
N_ANCILLA = N_ANCILLA_PER_BUS * N_BUS
SIZE = N_LOCS * N_BUS * N_TIMES
INDEX_STATION = 7

# Функция преобразования тензорных индексов в индексы QUBO
def tensor_to_qubo(location_index, time_index, bus_index):
    return N_TIMES * N_BUS * location_index + N_BUS * time_index + bus_index

BUS_RANGE = range(N_BUS)
TIME_RANGE = range(N_TIMES)
LOC_RANGE = range(N_LOCS)
CROSSROADS_RANGE = range(N_CROSSROADS)
SIGHTS_RANGE = range(N_CROSSROADS, N_LOCS)
ANCILLA_RANGE = range(N_ANCILLA_PER_BUS)

```

### Step 3: Building a QUBO matrix
```python
from itertools import product

QQ = np.zeros((7, SIZE + N_ANCILLA, SIZE + N_ANCILLA))

# Стоимость
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE[:-1]):
    for location_index_1, location_index_2, weight in edges:
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index + 1, bus_index)
        QQ[0, qubo_index_1, qubo_index_2] += weight

# Ограничения
# Первое ограничение: посещение каждой достопримечательности
for location_index in SIGHTS_RANGE:
    for bus_index_1, bus_index_2 in product(BUS_RANGE, repeat=2):
        for time_index_1, time_index_2 in product(TIME_RANGE, repeat=2):
            qubo_index_1 = tensor_to_qubo(location_index, time_index_1, bus_index_1)
            qubo_index_2 = tensor_to_qubo(location_index, time_index_2, bus_index_2)
            QQ[1, qubo_index_1, qubo_index_2] += 1
    for bus_index, time_index in product(BUS_RANGE, TIME_RANGE):
        qubo_index = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[1, qubo_index, qubo_index] -= 2

# Второе ограничение: один автобус на каждом узле в каждый момент времени
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE):
    for location_index_1, location_index_2 in product(LOC_RANGE, repeat=2):
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index, bus_index)
        QQ[2, qubo_index_1, qubo_index_2] += 1
    for location_index in LOC_RANGE:
        qubo_index = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[2, qubo_index, qubo_index] -= 2

# Третье ограничение: запрещенные переходы
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE[:-1]):
    for location_index_1, location_index_2 in no_edges:
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index + 1, bus_index)
        QQ[3, qubo_index_1, qubo_index_2] += 1

# Четвертое ограничение: ограничение на перекрестках
for location_index, time_index in product(CROSSROADS_RANGE, TIME_RANGE):
    for bus_index_1 in range(N_BUS):
        for bus_index_2 in range(bus_index_1 + 1, N_BUS):
            qubo_index_1 = tensor_to_qubo(location_index, time_index, bus_index_1)
            qubo_index_2 = tensor_to_qubo(location_index, time_index, bus_index_2)
            QQ[4, qubo_index_1, qubo_index_2] += 2

```
### Step 4: Start optimization and save the results

```python
weights = [0, 0, 1, 1, 0, 0, 0]
Q = sum(weights[i] * QQ[i] for i in range(7))

Q_triu = np.triu(Q + Q.T - np.diag(np.diag(Q)))
sol = pq.solve(Q_triu, number_of_runs=2, number_of_steps=100_000, dt=100, return_samples=False, verbose=10, gpu=True)

x = sol.vector
print("Количество активных переменных:", sum(x))
np.save("/workspace/kirill/res.npy", np.array(x))
```

Этот код выполняет загрузку данных, настройку параметров задачи, создание QUBO-матрицы и запуск оптимизации. Результаты сохраняются в файл res.npy.
