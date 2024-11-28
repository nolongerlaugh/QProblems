# Formation of an investment portfolio

## Description of the solution

This solution implements the Quadratic Unlimited Binary Optimization (QUBO) model for portfolio optimization. The code calculates the expected return, the covariance of asset returns, and then uses the QUBO formulation to optimize the allocation of capital between assets based on specified risk and return criteria. The goal is to optimally distribute capital among assets using QUBO methods

For more information about converting a goal into a loss function, see the presentation


The output of the program is a vector of integer variables that describes the number of shares purchased.

## Files

- `task-1-stocks.csv': A CSV file with historical stock prices for each asset.
  
## Quick start

### Step 1: Uploading and processing data

```python
import pandas as pd
import numpy as np

B = 1000000  # Бюджет или начальный капитал
p = pd.read_csv("task-1-stocks.csv")  # Загрузка исторических цен акций
N = len(p.columns)  # Количество активов
T = len(p)  # Количество временных периодов

# Предобработка данных
def r(p):
    return p.pct_change().dropna()

def mu(r):
    return r.mean()

def sigma(r):
    return r.cov()

p_s = p / B
r_s = r(p_s)
mu_s = mu(r_s)
sigma_s = sigma(r_s)
```
### Step 2: Generating the transformation matrix and parameters

```python
n_max = np.floor(B / p.iloc[0, :])  # Максимальное количество единиц каждого актива
d = np.array(np.ceil(np.log2(n_max)), dtype=int)

def c(d):
    dim = np.sum(d + 1)
    C = np.zeros((N, dim))
    k = 0
    for i in range(N):
        C[i, k: k + d[i]] = 1 << np.arange(d[i])
        C[i, k + d[i]] = n_max[i] + 1 - (1 << (d[i] - 1))
        k += d[i] + 1
    return C

C = c(d)
```

### Step 3: Calculating the QUBO Matrix

```python
def calculate_qubo_matrix(mu_ss, P_ss, sigma_ss, q, lambd, K=30, floor=True):
    n = len(mu_ss)
    Q = np.zeros((n + K, n + K))

    # Линейные коэффициенты: mu^T * b
    for i in range(n):
        Q[i, i] -= mu_ss[i]

    # Квадратичные коэффициенты: -q * b^T * sigma * b
    Q[:n, :n] -= q * np.array(sigma_ss)

    # Штрафные коэффициенты
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

params = {'q': 0.3, 'lambd': 136, 'p': 0.3148867250518056}
q = params['q']
K = 40
lambd = params['lambd'] / (1 << K)  # Коэффициент штрафа
floor = True

mu_ss = C.T @ mu_s
sigma_ss = C.T @ sigma_s @ C
P_ss = C.T @ np.floor(p_s.iloc[0, :] * (1 << K))
    
Q = calculate_qubo_matrix(mu_ss, P_ss, sigma_ss, q, lambd, K, floor)
```

### Step 4: Start optimization and evaluate the result
```python
print('Запуск оптимизации')
a = -np.triu(Q + Q.T - np.diag(np.diag(Q)))
sol = pq.solve(a, number_of_runs=2, number_of_steps=1000, return_samples=False, verbose=10, gpu=True, seed=239)
print('Результаты:')

x = C @ (sol.vector[:-K])
print('Потратили: ', x @ p.iloc[0, :])
print('Общий доход: ', float(x @ p.iloc[-1, :] - x @ p.iloc[0, :]))
print('Средний доход: ', float(R(x, p).iloc[0]))
print('Риск: ', float(Sigma(x, p).iloc[0]))
```

### Functions for calculating portfolio metrics

```python
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
```


