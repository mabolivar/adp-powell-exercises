import numpy as np
import pandas as pd

S = range(4)  # States
R = np.array([0, 10, 15, 20], ndmin=1)
P = np.array([[0.7, 0.3, 0, 0],
              [0, 0.8, 0.2, 0],
              [0, 0, 0.9, 0.1],
              [1, 0, 0, 0]])
A = {0: [0], 1: [0, 1], 2: [0, 1], 3: [1]}
M = 999

pd.set_option('display.max_rows', 500)


def max_norm(values):
    return np.max(np.abs(values))


def dict_to_df(d, colnames=None):
    df = pd.Series(d).reset_index()
    df.columns.values[-1] = 'value'
    print(colnames if colnames else df.columns.values.tolist())
    df.columns = colnames if colnames else df.columns.values.tolist()
    return df


def reward(s, a):
    return a * R[s]


def get_transition_matrix(policy, matrix):
    action_vector = np.array([1, 0 , 0 ,0], ndmin = 2)
    action_boolean = [i == 1 for i in policy]
    transition_matrix = np.copy(matrix)
    transition_matrix[action_boolean,] = action_vector
    return transition_matrix


def value_iteration(states, p, r, iterations, v_0={i: 0 for i in S}, discount_rate=0.8, epsilon=0.0005):
    v = {(0, s): v_0.get(s) for s in states}  # Value function (Iteration, state)
    x = {(0, s): v_0.get(s) for s in states}  # Optimal action (Iteration, state)
    v[(0, max(states))] = r[max(states)]
    n = 1
    while n < iterations:
        for s in states:
            for a in A[s]:
                future_contribution = discount_rate * (
                            a * v[(n - 1, 1)] + (1 - a) * sum(p[s, t] * v[(n - 1, t)] for t in states))
                current_contribution = reward(s, a)
                action_contribution = current_contribution + future_contribution
                if v.get((n, s), -M) < action_contribution:
                    v[(n, s)] = action_contribution
                    x[(n, s)] = a

        if max_norm([v[(n, s)] - v[(n - 1, s)] for s in states]) < epsilon * (1 - discount_rate) / (2 * discount_rate):
            break
        n += 1

    return (dict_to_df(v, colnames=['iteration', 'state', 'value']),
            dict_to_df(x, colnames=['iteration', 'state', 'action']))


def policy_iteration(states, transition_matrix, contribution_matrix, iterations, discount_rate=0.8, epsilon=0.005):
    v = {0: np.zeros((2, 1))}
    policies, p, c = {}, {}, {}
    policies[0] = [0, 0, 0, 1]
    to_iterate = [[0, i, j, 1] for i in A[1] for j in A[2]]
    num_states = len(states)
    n = 1
    while n < iterations:
        # Step 1 - Compute one-step transition matrix and contribution vector
        p[n-1] = get_transition_matrix(policies[n-1], transition_matrix)
        c[n-1] = (contribution_matrix * policies[n-1]).reshape(num_states, 1)

        inv = np.linalg.inv(np.identity(num_states) - discount_rate * p[n-1])
        v[n] = inv.dot(c[n-1])

        # Policy improvement
        v_tmp = [0 for i in states]
        policies[n] = [-1 for i in states]
        for policy in to_iterate:
            p_tmp = get_transition_matrix(policy, transition_matrix)
            c_tmp = (contribution_matrix * policy).reshape(num_states, 1)
            value = c_tmp + discount_rate * p_tmp.dot(v[n])
            for s in states:
                if v_tmp[s] <= value[s]:
                    v_tmp[s] = value[s]
                    policies[n][s] = policy[s]
        if policies[n] == policies[n-1]:
            break

        n += 1

    return dict_to_df(v, colnames=['iteration', 'value']), dict_to_df(policies, colnames=['iteration', 'action'])


def value_policy_iteration(states, transition_matrix, contribution_matrix,
                           iterations, v_0={i: 0 for i in S}, discount_rate=0.8, epsilon=0.0005):
    v = {0: np.zeros((4, 1))}
    to_iterate = [[0, i, j, 1] for i in A[1] for j in A[2]]
    num_states = len(states)
    v[(0, max(states))] = contribution_matrix[max(states)]
    n = 1
    policies = {}
    p, c, u = {}, {}, {}
    while True:
        v_tmp = [0 for i in states]
        policies[n] = [-1 for i in states]
        for policy in to_iterate:
            p_tmp = get_transition_matrix(policy, transition_matrix)
            c_tmp = (contribution_matrix * policy).reshape(num_states, 1)
            value = c_tmp + discount_rate * p_tmp.dot(v[n-1])
            for s in states:
                if v_tmp[s] <= value[s]:
                    v_tmp[s] = value[s]
                    policies[n][s] = policy[s]
        # Step 2
        p[n] = get_transition_matrix(policies[n], transition_matrix)
        c[n] = (contribution_matrix * policies[n]).reshape(num_states, 1)
        m = 0
        u[(n, m)] = c[n] + discount_rate * p[n].dot(v[n-1])

        if max_norm(u[(n, 0)] - v[n-1]) < epsilon * (1 - discount_rate) / (2 * discount_rate):
            break

        while m < iterations:
            u[(n, m + 1)] = c[n] + discount_rate * p[n].dot(u[(n, m)])
            m += 1
        v[n] = u[(n, iterations)]
        n += 1

    return dict_to_df(v, colnames=['iteration', 'value']), dict_to_df(policies, colnames=['iteration', 'action'])


if __name__ == '__main__':
    values, actions = value_iteration(S, P, R, iterations=100)
    solution = pd.merge(left=actions, right=values, left_on=['iteration', 'state'], right_on=['iteration', 'state'])
    print(solution)

    values, actions = policy_iteration(S, P, R, iterations=10)
    solution = pd.merge(left=actions, right=values, left_on=['iteration'], right_on=['iteration'])
    print(solution)

    values, actions = value_policy_iteration(S, P, R, iterations=10)
    solution = pd.merge(left=actions, right=values, left_on=['iteration'], right_on=['iteration'])
    print(solution)
