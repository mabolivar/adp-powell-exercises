import time
import numpy as np
from collections import defaultdict

NUM_TASKS = 10
BUDGET = 1000
STEPSIZE = 100
ACTIONS = [i for i in range(BUDGET + STEPSIZE) if i % STEPSIZE == 0]
R = {i: 0 for i in range(NUM_TASKS)}    # Remaining resource
V = {i: 0 for i in range(NUM_TASKS)}    # Value of having R_t resource remaining to allocate to task t and later tasks (optimal)
I = {i: 0 for i in range(NUM_TASKS)}    # Investment on each task


def contribution(task, spending):
    return (spending ** (1/4)) if task == 4 else np.sqrt(spending)


def transition(budget, spending):
    return budget - spending


def get_investment(task, budget, actions):
    best_contribution = 0
    best_spending = 0
    available_actions = [a for a in actions if a <= budget]

    for a in available_actions:
        future_contribution, _, _ = get_investment(task + 1, transition(budget, a), available_actions) \
            if task < NUM_TASKS - 1 else (0, None, None)
        action_contribution = contribution(task, a) + future_contribution
        if best_contribution < action_contribution:
            best_contribution = action_contribution
            best_spending = a

    return best_contribution, best_spending, budget


def solver(num_tasks, budget, actions):
    R[-1] = budget
    I[-1] = 0
    for task in range(num_tasks):
        prev_task = task - 1
        V[task], I[task], R[task] = get_investment(task, transition(R[prev_task], I[prev_task]), actions)

    investment_plan = {i: I[i] for i in range(num_tasks)}

    return V[0], investment_plan


if __name__ == "__main__":
    now = time.strftime('%d/%m/%Y %H:%M:%S')
    obj_fun, investment = solver(NUM_TASKS, BUDGET, ACTIONS)
    print("--------------------------")
    print("Current time:\t" + now)
    print("--------------------------")
    print('Objective function:\t' + str(obj_fun))
    print("Contributions:\t" + str({k: contribution(k, v) for k, v in investment.items()}))
    print('Investment:\t' + str(investment))
    print("--------------------------")
    print("Task\tValue\tSpending\tBudget")
    for task in range(NUM_TASKS):
        print(str(task) + "\t" + str(V[task]) + "\t" + str(I[task]) + "\t" + str(R[task]))
