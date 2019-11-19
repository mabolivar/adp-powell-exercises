from collections import defaultdict
import pandas as pd

S = [[i, j] for i in range(2) for j in range(2)]  # State space [1/0, 1/0] = is_free/non_free, non_parked/parked
A = [0, 1]  # Action space
N = range(1, 51)   # Parking spots
M = 999     # Large value

pd.set_option('display.max_rows', 500)


def dict_to_df(d, colnames=None):
    df = pd.Series(d).reset_index()
    df.columns.values[-1] = 'value'
    print(colnames if colnames else df.columns.values.tolist())
    df.columns = colnames if colnames else df.columns.values.tolist()
    return df


def transition(p, a):
    return p - a


def cost(s, a, n):
    parking_cost = 2 * n + 8 * (len(N) - n)
    return parking_cost * a


def get_solution(parking_spots, free_prob=[0.6, 0.4]):
    v = defaultdict()   # (spot, is_free, parked)
    len_spots = len(parking_spots)
    x = {(n, f, p): 0 for n in parking_spots for f, p in S}
    for f, p in S:
        v[(len_spots + 1, f, p)] = 30 + 2 * len_spots if p == 1 else 0

    for n in reversed(parking_spots):
        for f, p in S:
            action_space = [a for a in A if a <= p and a <= f]
            for a in action_space:
                future_cost = sum(free_prob[g] * v[(n+1, g, transition(p, a))] for g in [0, 1])
                current_cost = cost(p, a, n)
                if v.get((n, f, p), M) > a * current_cost + (1-a) * future_cost:
                    v[(n, f, p)] = current_cost + future_cost
                    x[(n, f, p)] = a

    return dict_to_df(v, ['spot', 'free', 'state', 'value']), dict_to_df(x, ['spot', 'free', 'state', 'action'])


if __name__ == '__main__':
    values, actions = get_solution(N)
    solution = pd.merge(left=actions, right=values, how='right',
                        left_on=['spot', 'free', 'state'], right_on=['spot', 'free', 'state'])
    solution['cost'] = cost(0, 1, solution.spot)
    print(solution[solution.state == 1])
