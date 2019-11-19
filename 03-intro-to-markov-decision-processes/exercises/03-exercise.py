
import numpy as np
import pandas as pd

import matplotlib
from plotnine import *

pd.set_option('display.max_rows', 500)
matplotlib.use('TkAgg')

S = [0, 1]
P = np.array([[0.7, 0.3], [0.05, 0.95]])    # One-step transition matrix
C = np.array([[10, 30], [20, 5]])   # Contribution from each transition


def max_norm(values):
    return np.max(np.abs(values))


def dict_to_df(d, colnames=None):
    df = pd.Series(d).reset_index()
    df.columns.values[-1] = 'value'
    print(colnames if colnames else df.columns.values.tolist())
    df.columns = colnames if colnames else df.columns.values.tolist()
    return df


def value_iteration(states, p, c, iterations, v_0={0: 0, 1: 0}, discount_rate=0.8, epsilon=0.0005):
    v = {(0, s): v_0.get(s) for s in states}
    n = 1
    while n < iterations:
        for s in states:
            v[(n, s)] = sum(p[s, t] * c[s, t] for t in states) + discount_rate * sum(p[s, t] * v[(n-1, t)] for t in states)

        if max_norm([v[(n, s)] - v[(n - 1, s)] for s in states]) < epsilon * (1 - discount_rate) / (2*discount_rate):
            break
        n += 1

    return dict_to_df(v, colnames=['iteration', 'state', 'value'])


def plot_evolution(df, state, file_name='figures/test.png'):
    to_plot = df[df.state == state]

    p = (ggplot(data=to_plot) +
         geom_point(aes(x='iteration', y='value')) +
         scale_x_continuous(expand=(0, 0)) +
         scale_y_continuous(expand=(0, 0))
         )
    p.save(filename=file_name)


if __name__ == '__main__':
    solution = value_iteration(S, P, C, 50, v_0={0: 100, 1: 0})
    print(solution)

    plot_evolution(solution, state=0, file_name='figures/03c-state_0_v_mixed.png')




