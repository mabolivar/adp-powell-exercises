# Some illustrative models
------------------------------------------------

This folder contains the scripts that implements the algorithms/pseudo-code presented in chapter 2 and solution of some exercises. 

## Content

### 01-deterministic-problems

+ `01-the-shortest-path-problem.py` contains an implementation of the algorithm presented in section 2.1.1
+ `02-the-discrete-budgeting-problem-forward.py` solves the discrete budgeting problem using a forward recursive approach.
+ `02-the-discrete-budgeting-problem-backward.py` implements a backward algorithm to solve the problem. 

### 02-stochastic-problems

+ `05-the-asset-acquisition-problem-i.py` presents the solution for the asset acquisition problem. It includes a simulation function that allows to test optimal policy results. The folder `figures` contains an example of an optimal policy and the behavior of the value function. 

## Comments

### 01-deterministic-problems

+ The implementation presented in `02-the-discrete-budgeting-problem-forward.py` is inefficient due to it computes multiple times the same values for each task. But is a good example of a recursive implementation that could be improved.
+ `02-the-discrete-budgeting-problem-backward.py` takes advantage of solving the problem by task layer however it might be improved by modeling the problem as a network (thinking in nodes instead of layers) and using the implementation in `01-the-shortest-path-problem.py`.

### 02-stochastic-problems

+ `05-the-asset-acquisition-problem-i.py` is an interesting problem and can be become more complex by including new benefits or penalizations. The implementation allows testing different demand distributions and their impact in multiple simulations. (Worth writing a short article about this problem and solution)



