# ABOUT
This is a python project for testing ideas regarding game tree search.
The game used is connect-four for simplicity.

# APPROACH
The search algorithm is based on a probabilistic best-first-search (different from mcts and minimax variants).
It applys Iterative Deepening and selectivly searches child nodes that look the most promising to change the minimax value of the current node.
The idea is that the frequency of a child node being searched shall be proportional to the probability of that child beeing better than all other children.
It uses discrete value distributions instead of scalars for (HEURISTIC-)evalution.
A simple (slow!!) neural network is implemented (with for python-loops xD) that can be trained to predict a value probability distribution.
The training.py skript runs selfplay games to generate data, trains the networks, runs validation matches and stores the networks in a folder.

# USAGE
With "python3 main.py" you can play against the latest (best) neural-net if available.
With "python3 training.py" you can train neural networks.

# Currents Project State:
    - semi-active

# HELP
(pls share bugfixes, ideas or recommendations regarding this project)



