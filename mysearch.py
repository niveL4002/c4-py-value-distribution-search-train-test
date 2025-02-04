import c4_board

import math
import time
import random
import itertools
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Callable, Any

class Solution(IntEnum):
    HEURISTIC = auto()
    WIN = auto()
    DRAW = auto()
    LOSS = auto()

    def __str__(self) -> str:
        return self.name

    def flip(self) -> Any:
        if self == Solution.HEURISTIC or self == Solution.DRAW:
            return self
        elif self == Solution.WIN:
            return Solution.LOSS
        return Solution.WIN

    def get_solved_D(self, D_len: int) -> [float]:
        assert self != Solution.HEURISTIC
        assert D_len % 2 == 1
        D = [0] * D_len
        if self == Solution.WIN:
            D[-1] = 1
        elif self == Solution.LOSS:
            D[0] = 1
        else:
            D[D_len // 2] = 1
        return D

# evaluation in [0, 1]
class Node:
    def __init__(self, solution: Solution = Solution.HEURISTIC,
                 D: int = None, nodes: int = 0):
        if D is not None:
            assert_valid_D(D)
        self.D: [float] = D
        self.solution = solution
        self.nodes = nodes

    def __str__(self) -> str:
        res = ""
        if self.D is not None:
            res += f"D = " + get_D_str(self.D)
        else:
            res += f"D = None : mean={self.get_mean()}"
        res += f"\nsolution = {self.solution}, nodes = {self.nodes}\n"
        return res

    def assert_valid_heuristic(self):
        assert self.solution == Solution.HEURISTIC
        if self.D is not None:
            assert_valid_D(self.D)
        return True

    def get_mean(self) -> float:
        if self.solution == Solution.HEURISTIC:
            assert self.assert_valid_heuristic()
            return get_D_mean(self.D)
        elif self.solution == Solution.WIN:
            return 1
        elif self.solution == Solution.LOSS:
            return 0
        else:
            assert self.solution == Solution.DRAW
            return 0.5

    # update nodes eval, arguments are interpreted from opponents perspective
    def update_heuristic_eval(self, heuristic_children: [Any], any_draw: int):
        assert self.assert_valid_heuristic()
        assert len(heuristic_children) > 0
        assert all(c.assert_valid_heuristic() for c in heuristic_children)
        D_len: int = len(heuristic_children[0].D)
        assert all(len(c.D) == D_len for c in heuristic_children)
        if self.D:
            assert len(self.D) == D_len
        else:
            self.D = [0] * D_len
        n = len(heuristic_children)
        sums = [0] * n
        smaller_equals = 0
        for i in range(D_len):
            self.D[i] = -smaller_equals
            smaller_equals = 1
            for j, child in enumerate(heuristic_children):
                sums[j] += child.D[D_len - 1 - i]
                smaller_equals *= sums[j]
            # propability of eval less than draw is 0 if there is a draw-move 
            if any_draw and (i < (D_len // 2)):
                smaller_equals = 0
            self.D[i] += smaller_equals
        assert self.assert_valid_heuristic()
        return

    # update nodes eval, arguments are interpreted from opponents perspective
    def new_update_heuristic_eval(self, heuristic_children: [Any], any_draw: int):
        assert self.assert_valid_heuristic()
        assert len(heuristic_children) > 0
        assert all(c.assert_valid_heuristic() for c in heuristic_children)
        D_len: int = len(heuristic_children[0].D)
        assert all(len(c.D) == D_len for c in heuristic_children)
        if self.D:
            assert len(self.D) == D_len
        else:
            self.D = [0] * D_len
        n = len(heuristic_children)
        weights = get_selection_weights(heuristic_children)
        for i in range(D_len):
            # propability of eval less than draw is 0 if there is a draw-move 
            if any_draw and (i < (D_len // 2)):
                self.D[i] = 0
                continue
            self.D[i] = sum(c.D[-i-1] * w for c, w in zip(heuristic_children, weights))
        s = sum(self.D)
        for i in range(D_len):
            self.D[i] /= s
        return

    def get_D(self, D_len: int) -> [float]:
        if self.solution == Solution.HEURISTIC:
            assert D_len == len(self.D)
            return self.D
        return self.solution.get_solved_D(D_len)

def assert_valid_D(D: [float]) -> bool:
    assert len(D) % 2 == 1
    assert all(v >= 0 for v in D)
    assert abs(sum(D) - 1) < 1e-4
    return True

def get_D_mean(D: [float]) -> float:
    return sum((i + 0.5) * v for i, v in enumerate(D)) / len(D)

def get_D_str(D: [float]) -> str:
    return "[" + ", ".join(f"{x:6<.3f}" for x in D) + f"] : mean={get_D_mean(D):.3}"

def uniform_evaluation(b: int, ob: int, D_len: int) -> [float]:
    return [1 / D_len for _ in range(D_len)]

def random_evaluation(b: int, ob: int, D_len: int) -> [float]:
    D = [random.random() for _ in range(D_len)]
    s = sum(D)
    return [d / s for d in D]

@dataclass
class EpochStats:
    def __init__(self):
        self.energy: int = 0
        self.visits_each_depth: [int] = [0]*42
        self.expansions_each_depth: [int] = [0]*42
        self.leafs_each_depth: [int] = [0]*42
        self.terminals_each_depth: [int] = [0]*42
        self.time_s: float = 0

@dataclass
class Stats:
    def __init__(self):
        self.each_epoch: [EpochStats] = []
        self.total_time_s: float = 0

    def __str__(self) -> str:
        res =  "epoch | energy    | time_s   | visits     | expansions | terminal | win | draw | loss\n"
        res += "      | visits each depth\n"
        res += "      | expansions each depth\n"
        res += "      | branching factors each depth\n"
        res += "      | leafs each depth\n"
        res += "      | terminals each depth\n"
        for i, x in enumerate(self.each_epoch):
            res += "-"*100 + "\n"
            res += f"{i+1:>2}    | {x.energy:>9} | {x.time_s:<08.3}"
            res += f" | {sum(x.visits_each_depth):>10}"
            res += f" | {sum(x.expansions_each_depth):>10}\n"
            j = max((i+1 for i, v in enumerate(x.visits_each_depth) if v), default=0)
            res += f"      | {[1] + x.visits_each_depth[:j]}\n"
            j = max((i+1 for i, v in enumerate(x.expansions_each_depth) if v), default=0)
            res += f"      | {[0] + x.expansions_each_depth[:j]}\n"
            bf = [b / a for a, b in itertools.pairwise([1] + x.visits_each_depth) \
                  if a > 0]
            res += f"      | {[round(a, 4) for a in bf]}\n"
            j = max((i+1 for i, v in enumerate(x.leafs_each_depth) if v), default=0)
            res += f"      | {[0] + x.leafs_each_depth[:j]}\n"
            j = max((i+1 for i, v in enumerate(x.terminals_each_depth) if v), default=0)
            res += f"      | {[0] + x.terminals_each_depth[:j]}\n"
        res += "-"*100 + "\n"
        res += f"total time: {self.total_time_s:.4} sec, "
        epoch_time: float = sum(x.time_s for x in self.each_epoch)
        total_visits: int = sum(sum(x.visits_each_depth) for x in self.each_epoch)
        res += f"speed: {round(total_visits / self.total_time_s)} visits / sec\n"
        res += "-"*100 + "\n"
        return res

@dataclass
class Params:
    D_len: int = 7
    # len(heuristic_evaluation) == D_len
    heuristic_evaluation: Callable = uniform_evaluation
    evaluation_params: Any = 7  # arg to heuristic_evaluation()
    min_energy: int = 42
    initial_energy: int = 42
    energy_scale: float = 1.5
    expansions_threshold: int = math.inf
    max_epochs: int = math.inf
    def assert_valid(self):
        assert self.heuristic_evaluation is not None
        assert self.D_len % 2 == 1
        assert self.min_energy > 0
        assert self.max_epochs > 0
        assert self.energy_scale >= 1
        return True
    def __str__(self):
        res = "heuristic: " + self.heuristic_evaluation.__name__
        res += f", D_len: {self.D_len}, "
        res += f"min_energy: {self.min_energy}, "
        res += f"intial_energy: {self.initial_energy}, "
        res += f"energy_scale: {self.energy_scale}, "
        res += f"expansions_threshold: {self.expansions_threshold}, "
        res += f"max_epochs: {self.max_epochs}"
        return res

class Result:
    def __init__(self, root: Node, board: int, other_board: int, TT: {(int, int): Node}):
        self.root = root
        self.best_move: int = -1
        self.root_child_results: [(int, Node)] = []  # (move, eval)
        for move in move_order:
            if not c4_board.is_valid_move(board, other_board, move):
                continue
            new_board = c4_board.make_move(board, other_board, move)
            if c4_board.is_win(new_board):
                child = Node(Solution.LOSS)
                self.root_child_results.append((move, child))
                continue
            if c4_board.get_move_count(other_board, new_board) == 42:
                child = Node(Solution.DRAW)
                self.root_child_results.append((move, child))
                continue
            child = TT.get((other_board, new_board))
            if child is None:
                child = TT.get(c4_board.get_mirrored(other_board, new_board))
            self.root_child_results.append((move, child))

        if root.solution == Solution.LOSS:
            self.best_move = max(self.root_child_results, key=lambda x: x[1].nodes)[0]
        elif any(child is not None for _, child in self.root_child_results):
            self.best_move = min(self.root_child_results,
                                 key=lambda x: x[1].get_mean() if x[1] else math.inf)[0]
        else:
            self.best_move = random.choice(self.root_child_results)[0]
            print("RANDOM MOVE\n"*30)
        return
    def calc_regret_of_move(self, move: int) -> float:
        if move == self.best_move:
            return 0
        best_val = 0
        move_val = 0
        for m, n in self.root_child_results:
            if m == self.best_move:
                best_val = n.get_mean()
            elif m == move:
                move_val = n.get_mean()
        return abs(best_val - move_val)

    def __str__(self) -> str:
        res = "   move  eval     nodes    solution\n"
        for i, (m, c) in enumerate(sorted(self.root_child_results, \
                                   key=lambda x: x[1].get_mean() if x[1] else math.inf)):
            res += f"{i+1}.  {m+1}   "
            res += (f"{float(1 - c.get_mean()):<05.3}" if c else "None ") +  "   "
            res += (f"{c.nodes:>07}" if c else "None ") +  "   "
            res += str(c.solution.flip() if c else None) + "\n"
        res += str(self.root)
        res += f"best_move: {self.best_move+1}\n"
        return res

def player(board: int, other_board: int, args: Any = None) -> int:
    if args is None:
        args = Params()
        args.max_epochs = 9
    TT = {}
    result, stats = search(board, other_board, args, TT)
    search_info(args, TT, result, stats)
    return result.best_move

def search_info(params: Params, TT: {(int, int): Node}, result: Result, stats: Stats):
    print("params:\n")
    print(params)
    print("stats:")
    print(stats)
    print("TT_len:", len(TT))
    print(result)
    return

def search(board: int, other_board: int, params: Params = None, \
           TT: {(int, int): Node} = None) -> (Result, Stats):
    assert params.assert_valid()
    start_time_s: float = time.perf_counter()
    if TT is None:
        TT = {}
    if params is None:
        params = Params()
    root = TT.get((board, other_board))
    if root is None:
        root = TT.get(c4_board.get_mirrored(board, other_board))
    if root is None:
        root = Node()
    stats = Stats()
    epoch: int = 1
    total_expansions: int = 0
    while epoch <= params.max_epochs and root.solution == Solution.HEURISTIC \
            and total_expansions < params.expansions_threshold:
        epoch_start_time_s: float = time.perf_counter()
        epoch_stats = EpochStats()
        epoch_stats.energy = max(params.min_energy,
            params.initial_energy if epoch == 1 \
            else math.ceil(stats.each_epoch[-1].energy * params.energy_scale))
        search_rec(board, other_board, root, epoch_stats.energy,
                   params, TT, epoch_stats)
        epoch_stats.time_s = time.perf_counter() - epoch_start_time_s
        stats.each_epoch.append(epoch_stats)
        total_expansions += sum(epoch_stats.expansions_each_depth)
        epoch += 1
    result = Result(root, board, other_board, TT)
    stats.total_time_s = time.perf_counter() - start_time_s
    return result, stats

move_order = [3, 2, 4, 1, 5, 0, 6]
def search_rec(board: int, other_board: int, node: Node, energy: int,
               params: Params, TT: {(int, int): Node}, stats: EpochStats, depth=0):
    # generate children and evaluate them,
    # terminate early if non-heuristic solution of node can be determined
    new_boards: [int] = []
    heuristic_children: [Node] = []
    any_draw: bool = False
    node.nodes = 0
    for move in move_order:
        if not c4_board.is_valid_move(board, other_board, move):
            continue
        stats.visits_each_depth[depth] += 1
        stats.leafs_each_depth[depth] += 1
        node.nodes += 1
        new_board = c4_board.make_move(board, other_board, move)
        if c4_board.is_win(new_board):
            node.solution = Solution.WIN
            stats.terminals_each_depth[depth] += 1
            return
        if c4_board.get_move_count(other_board, new_board) == 42:
            any_draw = True
            stats.terminals_each_depth[depth] += 1
            continue
        child = TT.get((other_board, new_board))
        # if child is None:
        #     child = TT.get(c4_board.get_mirrored(other_board, new_board))
        if child:
            node.nodes += child.nodes
            if child.solution == Solution.LOSS:
                node.solution = Solution.WIN
                node.D = None
                return
            if child.solution == Solution.DRAW:
                any_draw = True
            if child.solution != Solution.HEURISTIC:
                continue
        else:
            child = Node()
            child.D = params.heuristic_evaluation(other_board, new_board,
                                                  params.evaluation_params)
            assert len(child.D) == params.D_len
            assert child.assert_valid_heuristic()
            TT[(other_board, new_board)] = child
            stats.expansions_each_depth[depth] += 1
            energy -= 1
        new_boards.append(new_board)
        heuristic_children.append(child)

    if not heuristic_children:
        node.solution = Solution.DRAW if any_draw else Solution.LOSS
        node.D = None
        return
    if energy <= 0:
        node.new_update_heuristic_eval(heuristic_children, any_draw)
        return

    # search heuristic children recursively and try to solve them
    E = split_energy(heuristic_children, energy)
    child_order = sorted(zip(new_boards, heuristic_children, E),
                         key=lambda i: i[2], reverse=True)
    for new_board, child, child_energy in child_order:
        if child_energy <= 0:
            break
        stats.leafs_each_depth[depth] -= 1
        node.nodes -= child.nodes
        search_rec(other_board, new_board, child, child_energy,
                     params, TT, stats, depth + 1)
        node.nodes += child.nodes
        if child.solution == Solution.LOSS:
            node.solution = Solution.WIN
            node.D = None
            return
        if child.solution == Solution.DRAW:
            any_draw = True
    heuristic_children = [c for c in heuristic_children \
                          if c.solution == Solution.HEURISTIC]
    if not heuristic_children:
        node.solution = Solution.DRAW if any_draw else Solution.LOSS
        node.D = None
        return
    node.new_update_heuristic_eval(heuristic_children, any_draw)
    return

def split_energy(heuristic_children: [Node], energy: int) -> [int]:
    assert len(heuristic_children) > 0
    assert energy > 0
    assert all(child.assert_valid_heuristic() for child in heuristic_children)

    weights: [float] = get_selection_weights(heuristic_children)
    weights_sum = sum(weights)
    E: [int] = [math.floor(energy * w / weights_sum) for w in weights]
    assert sum(E) <= energy
    for _ in range(energy - sum(E)):
        E[thompson_select(heuristic_children)] += 1
    assert sum(E) > 0
    return E

# estimate for each child's evaluation distribution the 
# probability of being better than all other children
def get_selection_weights(heuristic_children: [Node]) -> [float]:
    n = len(heuristic_children)
    weights: [float] = [0] * n
    sums: [float] = [0] * n
    prods: [float] = [0] * n
    for i in range(len(heuristic_children[0].D) - 1, -1, -1):
        for j, child in enumerate(heuristic_children):
            sums[j] += child.D[i]
        product_except_self(sums, prods)
        for j, p in enumerate(prods):
            weights[j] += heuristic_children[j].D[i] * p
    return weights

def product_except_self(arr: [float], res: [float]):
    assert arr is not res
    assert len(arr) == len(res)
    res[0] = 1
    for i in range(1, len(arr)):
        res[i] = res[i-1] * arr[i - 1]
    prod: float = 1
    for i in range(len(arr) - 2, -1, -1):
        prod *= arr[i+1]
        res[i] *= prod
    return

def thompson_select(heuristic_children: [Node]) -> int:
    n = len(heuristic_children)
    return max(range(n), key=lambda i: -D_sample(heuristic_children[i].D))

# sample a scalar x from discrete evaluation distribution, 0 <= x <= 1
def D_sample(D: [float]) -> float:
    slot: int = random.choices(range(len(D)), weights=D, k=1)[0]
    return (slot + random.random()) / len(D)

def test():
    print("\nmysearch - test:   BEGIN")
    D_len: int = 7
    print(f"len(D) =", D_len)

    print("\nTEST update_heuristic_eval:")
    node = Node()
    node.D = random_evaluation(0, 0, D_len)
    node.assert_valid_heuristic()
    print("node:")
    print(node)
    children: [Node] = []
    n_children = 7
    for i in range(n_children):
        child = Node()
        child.D = uniform_evaluation(0, 0, D_len)
        children.append(child)
    c1 = Node()
    c1.D = [i*i for i in range(D_len)]
    c2 = Node()
    c2.D = [(D_len-i)**2 for i in range(D_len)]
    s1 = sum(c1.D)
    s2 = sum(c2.D)
    for i in range(D_len):
        c1.D[i] /= s1
        c2.D[i] /= s2
    children.append(c1)
    children.append(c2)

    print(f"means {n_children} children: ")
    print([round(child.get_mean(), 3) for child in children])
    print("updated node:")
    node.update_heuristic_eval(children, any_draw=False)
    print(node)
    node.assert_valid_heuristic()
    print("new updated node:")
    node.new_update_heuristic_eval(children, any_draw=False)
    print(node)
    node.assert_valid_heuristic()
    
    print("updated node: with draw")
    node.update_heuristic_eval(children, any_draw=True)
    print(node)
    node.assert_valid_heuristic()
    print("new updated node: with draw")
    node.new_update_heuristic_eval(children, any_draw=True)
    print(node)
    node.assert_valid_heuristic()


    print("\nTEST product_except_self:")
    arr = [random.randint(1, 5) for _ in range(6)]
    print(f"arr = {arr}")
    res = [0] * len(arr)
    product_except_self(arr, res)
    print(f"res = {res}")
    for i in range(len(arr)):
        prod = 1
        for j in range(len(arr)):
            if i != j:
                prod *= arr[j]
        assert prod == res[i]

    print("\nTEST get_selection_weights:")
    weights: [float] = get_selection_weights(children)
    print(f"weights: {[round(w, 3) for w in weights]}, sum: {sum(weights)}");

    print("\nTEST split_energy:")
    E: [float] = split_energy(children, 1000)
    print(f"E: {[round(e, 3) for e in E]}, sum: {sum(E)}");

    print("\nTEST get_mean:")
    node = Node()
    node.D = [1 / D_len] * D_len
    print(node)
    print(c1)
    print(c2)
    node.D = None
    node.solution = Solution.LOSS
    print(node)
    print("\nmysearch - test:   END\n")




