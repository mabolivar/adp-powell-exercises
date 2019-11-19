
import numpy as np
import pandas as pd

import matplotlib

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


def policy_iteration(states, transition_matrix, contribution_matrix, iterations, discount_rate=0.8, epsilon=0.005):
    v = {0: np.zeros((2, 1))}
    p, c = {}, {}
    pi = 0
    num_states = len(states)
    n = 1
    while n < iterations:
        p[n-1] = transition_matrix
        for s in states:
            c[n-1] = np.sum(contribution_matrix * p[n-1], axis=1, keepdims=True)

        inv = np.linalg.inv(np.identity(num_states) - discount_rate * p[n-1])
        v_tmp = inv.dot(c[n-1])

        # Policiy improvement
        v[n] = c[n-1] + discount_rate * p[n-1].dot(v_tmp)

        print(dict_to_df(v))

        if np.array_equal(v[n], v[n-1]):
            break
        n += 1

    return dict_to_df(v)


if __name__ == '__main__':
    policy_iteration_solution = policy_iteration(S, P, C, 50)
    print(policy_iteration_solution)
