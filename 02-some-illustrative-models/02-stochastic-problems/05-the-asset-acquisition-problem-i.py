import time
from collections import defaultdict
import numpy as np
import pandas as pd
import random

import matplotlib
matplotlib.use('TkAgg')
from plotnine import *

T = range(10)       # Time periods in the planning horizon
B = range(10)
ps = 2              # Purchase price of one resource unit
pp = 1              # Sell price of one resource unit
M = 99999           # Large value
discount_rate = 1   # Discount rate

D = {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.2, 5: 0.1}    # Demand probability distribution
#D = {0: 0.1, 1: 0.4, 2: 0.3, 3: 0.2}
#D = {0: 0.5, 5: 0.5}

R = {t: 0 for t in T}  # Assets on hand at time t before we make a new ordering decision, and beforewe have satisfied any demands arising in time interval t
X = defaultdict()  # Amount of product purchased at time t to be used during the time interval t + 1
V = defaultdict()  # Value of being in state R_t at each Value function V_t
P = defaultdict()


def demand_cum(u, demand_function):
    p_cum = 0
    for d, p in demand_function.items():
        p_cum += p
        if u < p_cum:
            return d

    return "Prob > 1"


def contribution(r, x, d):
    return ps * min(r, d) - pp * x - max(0, r - d) * 0.2 - max(0, d - r) * 2    # Selling reward - Purchasing cost - Inventory cost + Unsatisfied demand cost


def transition(r, x, d):
    return r - min(r, d) + x


def get_value(t):
    for r in B:
        for x in range(10):
            future_contribution = discount_rate * sum(p * V.get((t + 1, transition(r, x, d)), 0) for d, p in D.items())
            action_value = sum(p * contribution(r, x, d) for d, p in D.items()) + future_contribution
            #print({(r,x,d,p) : p * contribution(r, x, d) for d, p in D.items()})
            if V.get((t, r), -1 * M) < action_value:
                V[(t, r)] = action_value
                X[(t, r)] = x

    return None


def get_solution(action_dict, value_dict):
    x_df = pd.Series(action_dict).reset_index()
    x_df.columns = ['period', 'inventory', 'action']

    v_df = pd.Series(value_dict).reset_index()
    v_df.columns = ['period', 'inventory', 'V']
    v_df['V'] = [float(x) for x in np.round(v_df['V'], 0)]
    return x_df, v_df


def solver():
    for t in reversed(T):
        get_value(t)


def run_simulation(iterations, ini_inventory, debug=True):
    values = []
    for i in range(iterations):
        inventory = ini_inventory
        value_cum = 0
        if debug:
            print('==========================================')
            print('Time\tInventory\tAction\tDemand\tReward')
        for t in T:
            u = random.random()
            action = X[(t, inventory)]
            demand = demand_cum(u, D)
            reward = contribution(inventory, action, demand)
            if debug:
                print(str(t) + '\t' + str(inventory) + '\t' + str(action) + '\t' + str(demand) + '\t' + str(reward))

            inventory = transition(inventory, action, demand)
            value_cum += reward
        values.append(value_cum)
        if debug:
            print('==========================================')
            print('Iterations:\t' + str(i) +
                  '\tE(value):\t' + str(np.round(V[0, ini_inventory], 2)) +
                  '\tAvg(value):\t' + str(np.round(np.mean(values), 2)) +
                  '\tsd(value):\t' + str(np.round(np.std(values), 2)))

    return values


def save_policy(df, var_name, file_name):
    p = (ggplot(data=df) +
         geom_tile(aes(x='inventory', y='period', fill=var_name)) +
         geom_text(aes(x='inventory', y='period', label=var_name)) +
         scale_x_continuous(expand=(0, 0), breaks=B) +
         scale_y_continuous(expand=(0, 0), breaks=T)
         )
    p.save(filename=file_name)


if __name__ == '__main__':
    now = time.strftime('%d/%m/%Y %H:%M:%S')

    # Call solver and parse solutions
    solver()
    actions, values = get_solution(X, V)

    # Simulation
    random.seed(876)
    iterations = 20
    initial_inventory = 5
    value_iterations = run_simulation(iterations, initial_inventory)

    # Save policies images and value matrix
    save_policy(actions, 'action', 'figures/05-actions.png')
    save_policy(values, 'V', 'figures/05-values.png')

    print('\n')
    print('******************************************')
    print('Time:\t' + now)
    print('******************************************')
    print('Policy:')
    print(actions.pivot(index='period', columns='inventory', values='action'))
    print('******************************************')
    print('Simulation results:')
    print('Iterations:\t' + str(iterations) +
          '\tV(value):\t' + str(np.round(V[0, initial_inventory], 2)) +
          '\tAvg(value):\t' + str(np.round(np.mean(value_iterations), 2)) +
          '\tsd(value):\t' + str(np.round(np.std(value_iterations), 2)))
    print('******************************************' + '\n')

