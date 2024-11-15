import pandas as pd
import numpy as np
import sys
B = 1000000
p = pd.read_csv("task-1-stocks.csv")
N = len(p.columns)
T = len(p)


def r(p):
    return p.pct_change().dropna()
    # difference_df = (p.shift(-1) - p).iloc[:-1]
    # return difference_df / p.iloc[0]

def mu(r):
    return r.mean()

def sigma(r):
    return r.cov()

def R(x, p):
    ps = np.zeros(T)
    for i in range(len(ps)):
        ps[i] = x @ p.iloc[i, :]
    ps = pd.DataFrame(ps)
    rs = r(ps)
    mus = mu(rs)
    return mus

def Sigma(x, p):
    ps = np.zeros(T)
    for i in range(len(ps)):
        ps[i] = x @ p.iloc[i, :]
    ps = pd.DataFrame(ps)
    rs = r(ps)
    mus = mu(rs)
    Sigma = np.sum((rs - mus)**2) * N / (N - 1)
    Sigma = np.sqrt(Sigma)
    return Sigma

  
p = p.loc[:, ~(r(p) <= 0).all(axis=0)]
N = len(p.columns)
T = len(p)

p_s = p / B
r_s = r(p_s)
mu_s = mu(r_s)
sigma_s = sigma(r_s)

n_max = np.floor(B / p.iloc[0, :])

d = np.array(np.ceil(np.log2(n_max)), dtype = int)

def c(d):
    dim = np.sum(d + 1)
    C = np.zeros((N, dim))
    k = 0
    for i in range(N):
        C[i, k: k + d[i]] = 1 << np.arange(d[i])
        C[i, k + d[i]] = n_max[i] + 1 - (1 << (d[i] - 1))
        k += d[i] + 1
    return C



import numpy as np

def calculate_qubo_matrix(mu_ss, P_ss, sigma_ss, q, lambd, K = 30, floor = True):
    """
    Вычисляет коэффициенты QUBO матрицы для данной задачи.

    :param mu: Вектор ожидаемых доходностей (numpy array)
    :param P: Вектор вероятностей или весов (numpy array)
    :param sigma: Ковариационная матрица доходностей (numpy array)
    :param q: Коэффициент риска
    :param lambd: Коэффициент штрафа (lambda)
    :return: QUBO матрица (numpy array)
    """
    n = len(mu_ss)

    Q = np.zeros((n + K, n + K))

    # Часть 1: mu^T * b (линейные коэффициенты)
    for i in range(n):
        Q[i, i] += mu_ss[i] 

    # Часть 2: -q * b^T * sigma * b (квадратичные коэффициенты)
    Q[:n, :n] -= q * np.array(sigma_ss)

    # # Часть 3: -lambda * (P^T * b - 1)^2 (квадратичные и линейные штрафы)
    # # Раскрываем (P^T * b - 1)^2 = P^T * b * P^T * b - 2 * P^T * b + 1
    # for i in range(n):
    #     for j in range(n):
    #         Q[i, j] -= lambd * P_ss[i] * P_ss[j]
    #     Q[i, i] += 2 * lambd * P_ss[i]


    #Добавляем еще K кубитов
    P_ss_c = P_ss
    for i in range(n):
        Q[i, i] += lambd * (1 << K) * 2 * P_ss_c[i]
    for j in range(K):
        Q[n + j, n + j] += lambd * (1 << (K + j)) * 2
    
    for i in range(n):
        for j in range(n):
            Q[i, j] -= lambd * P_ss_c[i] * P_ss_c[j]
    
    for i in range(K):
        for j in range(K):
            Q[n + i, n + j] -= lambd * (1 << (i + j))
            
    for i in range(n):
        for j in range(K):
            Q[i, n + j] -= 2 * lambd * P_ss_c[i] * (1 << j)
    return Q

import pyqiopt as pq


######## Для  1 запуска ####
params = {'q': 0.3977928646223712, 'lambd': 379.0870163094632}
q = params['q']
K = 40
lambd = params['lambd'] / (1 << K)  # коэффициент штрафа
floor = True


C = c(d)
mu_ss = C.T @ mu_s
sigma_ss = C.T @ sigma_s @ C
#P_ss = C.T @ p_s.iloc[0, :]
P_ss = C.T @ np.floor(p_s.iloc[0, :] * (1 << K))

# print('Запуск')
# a = -np.triu(Q + Q.T - np.diag(np.diag(Q)))
# sol = pq.solve(a, number_of_runs=1, number_of_steps=100, return_samples=False, verbose=10, gpu=True)
# print('ok')
# #print(sol.vector, sol.objective)

# x = C @ (sol.vector[:-K])
# print('Потратили: ', x @ p.iloc[0, :])
# print('Общий доход: ', float((x @ p.iloc[-1, :] - x @ p.iloc[0, :])))
# print('Средний доход: ', float(R(x, p)).iloc[0])
# print('Риск: ', float(Sigma(x, p)).iloc[0])

######### Для подбора гиперпараметров ############

import random
import subprocess

def random_search(param_ranges, num_iterations, p, f = sys.stdout):
    best_params = None
    best_val = -float('inf')

    for i in range(num_iterations):
        
        params = {param: random.uniform(*param_ranges[param]) for param in param_ranges}

        q = params['q']  # коэффициент риска
        lambd =  params['lambd'] / (1 << (K))  # коэффициент штрафа
        Q = calculate_qubo_matrix(mu_ss, P_ss, sigma_ss, q, lambd, K)
        
        a = -np.triu(Q + Q.T - np.diag(np.diag(Q)))
        sol = pq.solve(a, number_of_runs=2, number_of_steps=10000, return_samples=False, verbose=50, gpu=True, dt=100, q=0)  

        x = C @ (sol.vector[:-K ])
        #x = C @ (sol.vector[:])
        
        # val = float(R(x, p).iloc[0])
        # risk = float(Sigma(x, p).iloc[0])
        summa = float(x @ p.iloc[0, :])

        val = x @ mu(r(p))
        risk = np.dot(x, sigma(r(p)) @ x) / sum(x)**2
        

        if risk <= 0.25 and val > best_val and summa <= B:
          
            best_val = val
            best_params = params
            print(x, file = f)
        
        print(f"Итерация {i + 1}/{num_iterations}, Параметры: {params}, Сумма: {summa}, Доход: {val * 10**4}, Риск: {risk}", file = f)

    return best_params, best_val

# Диапазоны значений для параметров
param_ranges = {
    'q': (0.0002, 0.5),
    'lambd': (100, 400),
}

# Количество итераций случайного поиска
num_iterations = 10

# Запуск случайного поиска
with open('/workspace/1/tunes_small_012.txt', 'w+') as f:
  best_params, best_loss = random_search(param_ranges, num_iterations, p, f)
  print(f"\nНаилучшие параметры: {best_params}", file=f)
  print(f"Наилучшее значение функции дохода: {best_loss * 10**4}", file=f)

