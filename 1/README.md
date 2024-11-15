# Формирование инвестиционного портфеля

## Описание решения

Это решение реализует модель квадратичной неограниченной бинарной оптимизации (QUBO) для оптимизации портфеля. Код вычисляет ожидаемую доходность, ковариацию доходностей активов и затем использует QUBO-формулировку для оптимизации распределения капитала между активами на основе заданных критериев риска и доходности. Цель — оптимально распределить капитал среди активов, используя методы QUBO

Подробее о преобразовании цели в функцию потерь смотреть в презентации


Выходом программы является вектор целочисленных переменных, который описывает количество купленных акций. 

## Файлы

- `task-1-stocks.csv`: CSV-файл с историческими ценами акций для каждого актива. 

## Быстрый старт

### Шаг 1: Загрузка и обработка данных

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
### Шаг 2: Генерация матрицы преобразования и параметров

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

### Шаг 3: Вычисление QUBO-матрицы

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

### Шаг 4: Запуск оптимизации и оценка результата
print('Запуск оптимизации')
a = -np.triu(Q + Q.T - np.diag(np.diag(Q)))
sol = pq.solve(a, number_of_runs=2, number_of_steps=1000, return_samples=False, verbose=10, gpu=True, seed=239)
print('Результаты:')

x = C @ (sol.vector[:-K])
print('Потратили: ', x @ p.iloc[0, :])
print('Общий доход: ', float(x @ p.iloc[-1, :] - x @ p.iloc[0, :]))
print('Средний доход: ', float(R(x, p).iloc[0]))
print('Риск: ', float(Sigma(x, p).iloc[0]))

### Функции для расчета метрик портфеля

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


