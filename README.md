# ABOUT
This is a python project for testing ideas regarding game tree search.
The game used is connect-four for simplicity.

# APPROACH
The search algorithm is based on a probabilistic best-first-search (different from mcts and minimax variants).
The correct minimax solution (WIN, DRAW, LOSS or HEURISTIC) is propagated from the leaf-nodes to the root.
It applies Iterative Deepening and selectivly searches child nodes that look the most promising to change the minimax value of the current node.
The idea is that the frequency of a child node being searched shall be proportional to the probability of that child beeing better than all other children.
It uses discrete value distributions instead of scalars for (HEURISTIC-)evalution.
A simple (slow!!) neural network is implemented (with for python-loops xD) that can be trained to predict a value probability distribution.
The training.py skript runs selfplay games to generate data, trains the networks, runs validation matches and stores the networks in a folder.

# USAGE
With "python3 main.py" you can play against the latest (best) neural-net(s) if available or let two of them play against each other.
signature: "python3 main.py [playerX] [playerO]"
where playerX or playerO can be one of the following:
    "h" (for human move input),
    "c" (for computer-player with latest net),
    a number in [0, latest net index] (computer-player with net corresponding to the number (see c4-train))

With "python3 training.py" you can train neural networks.

# HELP
Please share bugfixes, ideas or recommendations regarding this project.
If you have similiar projects please let me know.




