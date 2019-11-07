import time
import numpy as np
from collections import defaultdict

NUM_TASKS = 10
BUDGET = 1000
STEPSIZE = 100
ACTIONS = [i for i in range(BUDGET + STEPSIZE) if i % STEPSIZE == 0]
R = {i: 0 for i in range(NUM_TASKS)}    # Remaining resource
I = {(i, BUDGET - a): 0 for i in range(NUM_TASKS) for a in ACTIONS}    # Investment on each task
V = {(i, BUDGET - a): 0 for i in range(NUM_TASKS) for a in ACTIONS}    # Value of having R_t resource remaining to allocate to task t and later tasks (optimal)
P = defaultdict()  # Next state


def contribution(task, spending):
    return (spending ** (1/4)) if task == 4 else np.sqrt(spending)


def transition(budget, spending):
    return budget - spending


def get_investment(task, budget, actions):

    for b in budget:
        available_actions = [a for a in actions if a <= b]
        for a in available_actions:
            future_contribution = V.get((task + 1, b - a), 0)
            action_contribution = contribution(task, a) + future_contribution
            if V[(task, b)] < action_contribution:
                V[(task, b)] = action_contribution
                I[(task, b)] = a
                P[(task, b)] = (task + 1, b - a)

    return V, I, P


def get_path(node, predecesors):
    path = list()
    p = get_path(predecesors[node], predecesors) if node in predecesors.keys() else []
    path.append(node)
    return path + p


def solver(num_tasks, budget, actions):
    start = (0, budget)
    for task in reversed(range(0, num_tasks)):
        get_investment(task, actions, actions)

    path = get_path(start, P)
    investment_plan = {k: I[(k, v)] for k, v in path if k < num_tasks}

    return V[start], investment_plan, path


if __name__ == "__main__":
    now = time.strftime('%d/%m/%Y %H:%M:%S')
    obj_fun, investment, path = solver(NUM_TASKS, BUDGET, ACTIONS)
    print("--------------------------")
    print("Current time:\t" + now)
    print("--------------------------")
    print('Objective function:\t' + str(obj_fun))
    print("Contributions:\t" + str({k: contribution(k, v) for k, v in investment.items()}))
    print('Investment:\t' + str(investment))
    print("--------------------------")
    print("Task\tValue\tSpending\tBudget")
    for task, budget in path:
        if task in investment.keys():
            print(str(task) + "\t" + str(V[(task, budget)]) + "\t" + str(I[(task, budget)]) + "\t" + str(budget))