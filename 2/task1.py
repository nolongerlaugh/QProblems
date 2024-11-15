import pandas as pd
import numpy as np
import pyqiopt as pq

adj = pd.read_csv("/workspace/kirill/task-2-adjacency_matrix.csv")
nodes = pd.read_csv("/workspace/kirill/task-2-nodes.csv", names = ["name","n_people"])

nodes.columns = ["name", "n_people"]
nodes.sort_values(by='n_people', inplace=True)
nodes = nodes.reset_index(drop=True)

name_to_index = dict()
index_to_name = dict()
for i in nodes.index:
    index_to_name[i] = nodes['name'][i]
    name_to_index[nodes['name'][i]] = i
    
people_count = nodes.drop(columns=['name']).to_numpy().T[0]
N_LOCS = nodes.shape[0]
N_CROSSROADS = N_LOCS -  np.count_nonzero(nodes['n_people'])
N_BUS = 15
N_TIMES = 16

adj_ = adj.set_index("Unnamed: 0")
edges = list()
no_edges = list()
cost_matrix = np.zeros(shape=(N_LOCS,N_LOCS))
for i in range(N_LOCS):
    for j in range(i+1, N_LOCS):
        char = adj_[index_to_name[i]][index_to_name[j]]
        if char != '-':
            edges.append((i,j,int(char)))
        else:
            no_edges.append((i,j))
            no_edges.append((j,i))


#В задаче присутствует ограничение типа неравенства, поэтому нам понадобится несколько анцилл
#Смысл этих переменных станет яснее чуть позже
N_ANCILLA_PER_BUS = 4
N_ANCILLA = N_ANCILLA_PER_BUS * N_BUS

#Размер задачи (без анцилл)
SIZE = N_LOCS * N_BUS * N_TIMES 
#Для одного из ограничений нам понадобится индекс вокзала
INDEX_STATION = 7 

#Перевод из индексов тензора (индекс ноды, время, номер автобуса) в индекс в кубо матрице
def tensor_to_qubo(location_index, time_index, bus_index):
    return N_TIMES * N_BUS * location_index + N_BUS * time_index + bus_index

BUS_RANGE = range(0, N_BUS)
TIME_RANGE = range(0, N_TIMES)
LOC_RANGE = range(0, N_LOCS)
CROSSROADS_RANGE = range(0, N_CROSSROADS) #Первые 22 ноды - это перекрёстки
SIGHTS_RANGE = range(N_CROSSROADS, N_LOCS) #Все остальные ноды - достопримечательности
ANCILLA_RANGE = range(0, N_ANCILLA_PER_BUS) 

from itertools import product # Ну, поехали...
QQ = np.zeros(shape = (7,SIZE + N_ANCILLA, SIZE + N_ANCILLA))

#cost
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE[:-1]):
    for edge in edges:
        location_index_1, location_index_2, weight = edge
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index + 1, bus_index)
        QQ[0, qubo_index_1, qubo_index_2] += weight
        

#first
for location_index in SIGHTS_RANGE: #по всем достопримечательностям
    
    for bus_index_1, bus_index_2 in product(BUS_RANGE, repeat = 2):
        for time_index_1, time_index_2 in product(TIME_RANGE, repeat = 2):
            qubo_index_1 = tensor_to_qubo(location_index, time_index_1, bus_index_1)
            qubo_index_2 = tensor_to_qubo(location_index, time_index_2, bus_index_2)
            QQ[1,qubo_index_1, qubo_index_2] += 1
            
    for bus_index, time_index in product(BUS_RANGE, TIME_RANGE):
        qubo_index = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[1,qubo_index, qubo_index] -= 2

#second
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE):
    
    for location_index_1, location_index_2 in product(LOC_RANGE, repeat = 2):
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index, bus_index)
        QQ[2,qubo_index_1, qubo_index_2] += 1
            
    for location_index in LOC_RANGE:
        qubo_index = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[2,qubo_index, qubo_index] -= 2


#third
for bus_index, time_index in product(BUS_RANGE, TIME_RANGE[:-1]):
    for location_index_1, location_index_2 in no_edges:
        qubo_index_1 = tensor_to_qubo(location_index_1, time_index, bus_index)
        qubo_index_2 = tensor_to_qubo(location_index_2, time_index+1, bus_index)
        QQ[3,qubo_index_1, qubo_index_2] += 1

#4th
for location_index, time_index in product(CROSSROADS_RANGE, TIME_RANGE):
    
    for bus_index_1 in range(N_BUS):
        for bus_index_2 in range(bus_index_1 + 1, N_BUS):
            
            qubo_index_1 = tensor_to_qubo(location_index, time_index, bus_index_1)
            qubo_index_2 = tensor_to_qubo(location_index, time_index, bus_index_2)
            
            QQ[4,qubo_index_1, qubo_index_2] += 2      

#5th 
for bus_index in range(N_BUS):
    
    for location_index_1, location_index_2 in product(LOC_RANGE, repeat = 2):
        for time_index_1, time_index_2 in product(TIME_RANGE, repeat = 2):
            
            qubo_index_1 = tensor_to_qubo(location_index_1, time_index_1, bus_index)
            qubo_index_2 = tensor_to_qubo(location_index_2, time_index_2, bus_index)
            
            QQ[5,qubo_index_1, qubo_index_2] += (people_count[location_index_1] * people_count[location_index_2])
            
    for ancilla_this_bus_1, ancilla_this_bus_2 in product(ANCILLA_RANGE, repeat = 2):

        ancilla_index_1 = bus_index * N_ANCILLA_PER_BUS + ancilla_this_bus_1
        ancilla_index_2 = bus_index * N_ANCILLA_PER_BUS + ancilla_this_bus_2

        qubo_index_1 = SIZE + ancilla_index_1
        qubo_index_2 = SIZE + ancilla_index_2

        QQ[5,qubo_index_1, qubo_index_2] += 2 ** (ancilla_this_bus_1 + ancilla_this_bus_2)
            
    for location_index, time_index, ancilla_this_bus in product(LOC_RANGE, TIME_RANGE, ANCILLA_RANGE):
                
        ancilla_index = bus_index * N_ANCILLA_PER_BUS + ancilla_this_bus
        qubo_index_1 = SIZE + ancilla_index
        qubo_index_2 = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[5,qubo_index_1, qubo_index_2] += 2 * 2**ancilla_this_bus * people_count[location_index]
                
    for location_index, time_index in product(LOC_RANGE, TIME_RANGE):
        qubo_index = tensor_to_qubo(location_index, time_index, bus_index)
        QQ[5,qubo_index, qubo_index] += 2 * (-10) * people_count[location_index]
            
    for ancilla_this_bus in ANCILLA_RANGE:
        
        ancilla_index = bus_index * N_ANCILLA_PER_BUS + ancilla_this_bus
        qubo_index = SIZE + ancilla_index
        QQ[5,qubo_index, qubo_index] += 2 * (-10) * 2**ancilla_this_bus


#6th
for bus_index in BUS_RANGE:
    qubo_index_1 = tensor_to_qubo(INDEX_STATION, N_TIMES-1, bus_index)
    qubo_index_2 = tensor_to_qubo(INDEX_STATION, 0, bus_index)
    QQ[6, qubo_index_1, qubo_index_2] -= 1


weights = [0,0,1,1,0,0,0]

Q = np.zeros(shape = (SIZE + N_ANCILLA, SIZE + N_ANCILLA))
for i in range(7):
    Q += weights[i] * QQ[i]

Q_triu = np.triu(Q + Q.T - np.diag(np.diag(Q)))

sol = pq.solve(Q_triu, number_of_runs = 2, number_of_steps = 100_000, dt = 100, return_samples=False, verbose=10, gpu=True)
x = sol.vector
print(sum(x))

np.save("/workspace/kirill/res.npy", np.array(x))

