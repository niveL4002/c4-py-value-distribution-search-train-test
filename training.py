import c4_board
import mysearch
import brain

import time
import json
import random
from typing import Callable, Any
from pathlib import Path

train_data_path = "c4-train/"
meta_filepath = train_data_path + "meta.json"
def get_net_filepath(net_index) -> str:
    if net_index < 0: return None
    return train_data_path + "net" + str(net_index) + ".json"

# return -1 if not available
def get_latest_net_index() -> int:
    # read meta
    last_net_index = -1
    try:
        f = open(meta_filepath, "r")
    except FileNotFoundError:
        print(meta_filepath, "not found")
    else:
        print("read", meta_filepath, "...")
        with f:
            last_net_index = int(f.read(64))
    return last_net_index

def get_latest_net_filepath() -> str:
    i = get_latest_net_index()
    if i == -1:
        return None
    return get_net_filepath(i)

def main():
    print("training.py - main")
    mysearch.test()

    Path(train_data_path).mkdir(exist_ok=True)

    # params
    net_layer_sizes = [43, 60, 60, 5]
    data_epochs = 10
    games_per_epoch = 400
    train_epochs_per_data_epoch = 4
    validation_period_in_epochs = 1
    validation_match_game_pairs = 10
    datagen_min_nodes = 100
    batch_size = 30
    lr = 0.01
    mom = 0.9
    loss_function = distribution_loss
    d_loss_function = d_distribution_loss
    search_params = mysearch.Params()
    search_params.D_len = net_layer_sizes[-1]
    search_params.heuristic_evaluation = mysearch.uniform_evaluation
    search_params.evaluation_params = search_params.D_len
    search_params.expansions_threshold = 1000
    search_params.energy_scale = 1.5

    # read meta
    last_net_index = get_latest_net_index()
    if last_net_index != -1:
        print("last net index:", last_net_index)

    # get nn
    if last_net_index == -1:
        nn = brain.NN(L=net_layer_sizes)
    else:
        nn = brain.NN(filepath=get_net_filepath(last_net_index))
        search_params.evaluation_params = nn
        search_params.heuristic_evaluation = nn_evaluation
    print("NN:", nn)
    if last_net_index > 0:
        other_nn = brain.NN(filepath=get_net_filepath(last_net_index - 1))
        nn.compare_info(other_nn)
    # run validation match against previous versions
    # for i in range(last_net_index-1, -1, -1):
    #     old_nn = brain.NN(filepath=get_net_filepath(i))
    #     validation_match(nn, old_nn, validation_match_game_pairs, search_params)

    # training loop
    for data_epoch in range(1, data_epochs + 1):
        print(f"data epoch: {data_epoch}/{data_epochs}")
        data = generate_selfplay_data(search_params, \
                                      games_per_epoch, datagen_min_nodes, data_epoch)
        for nr_train_epoch in range(train_epochs_per_data_epoch):
            train_epoch(nn, data, batch_size, lr, mom, loss_function, d_loss_function, \
                        nr_train_epoch, data_epoch, print_duration_s=3)
        show_examples(nn, random.choices(data, k=20), loss_function)
        if data_epoch % validation_period_in_epochs and last_net_index >= 0: continue
        if last_net_index >= 0:
            old_nn = brain.NN(filepath=get_net_filepath(last_net_index))
            new_is_better = validation_match(nn, old_nn, validation_match_game_pairs,
                                             search_params)
            if not new_is_better: continue
        else:
            search_params.evaluation_params = nn
            search_params.heuristic_evaluation = nn_evaluation
        last_net_index += 1
        nn.save(get_net_filepath(last_net_index))
        with open(meta_filepath, "w") as f:
            f.write(str(last_net_index))
    return

def show_examples(nn: brain.NN, data: [(int, int, mysearch.Node)], loss_f: Callable):
    print(f"show {len(data)} examples:")
    print(f"NN: {nn}")
    total_loss: float = 0
    for i, (board, other_board, node) in enumerate(data):
        if random.randint(0, 1):
            board, other_board = c4_board.get_mirrored(board, other_board)
        print(f"\nexample {i+1}:")
        print(f"(nodes={node.nodes}, solution={node.solution})")
        print("board:")
        c4_board.print_board(board, other_board)
        I = c4_board.get_nn_input(board, other_board, with_side_to_play=True)
        print("nn input:")
        c4_board.print_nn_input(I)
        O = nn.forward(I)
        print("nn output          :", mysearch.get_D_str(O))
        PO = processed_nn_output(O)
        print("processed nn output:", mysearch.get_D_str(PO))
        EO = node.get_D(D_len=nn.L[-1])
        print("expected output    :", mysearch.get_D_str(EO))
        error = loss_f(PO, EO)
        print("processed output loss:", round(error, 3))
        error = loss_f(O, EO)
        print("loss:", round(error, 3))
        total_loss += error
    print(f"mean loss over {len(data)} examples: {round(total_loss / len(data), 3)}")
    return

def train_epoch(nn: brain.NN, game_data: [(int, int, mysearch.Node)],
                batch_size: int, lr: float, mom: float,
                loss_f: Callable, d_loss_f: Callable,
                nr_train_epoch: int, data_epoch: int, print_duration_s: float = 1):
    indices = list(range(2 * len(game_data)))
    random.shuffle(indices)
    batches, rest = divmod(len(indices), batch_size)
    a, b = divmod(rest, batches)
    batch_sizes = [batch_size + a + (b > i) for i in range(batches)]
    assert sum(batch_sizes) == len(indices)
    nn.zero_gradient()
    batch_index = 0
    loss = 0
    loss_count = 0
    total_time_start = print_time_start = time.perf_counter()
    print("data_epoch:", data_epoch, "train epoch:", nr_train_epoch,
          "data size:", len(indices), "batch_size;", batch_size, "lr:", lr, "mom", mom)
    for i in indices:
        if time.perf_counter() - print_time_start >= print_duration_s:
            print("data_epoch:", data_epoch, "train epoch:", nr_train_epoch, end=" ")
            print(f"batches: {batch_index + 1}/{batches} | ", \
                  f"time: {round(time.perf_counter() - total_time_start, 1)} | ",
                  f"mean loss over {loss_count:4} items: {round(loss / loss_count, 3)}")
            loss_count = 0
            loss = 0
            print_time_start = time.perf_counter()
        board, other_board, node = game_data[i % len(game_data)]
        if i >= len(game_data):
            board, other_board = c4_board.get_mirrored(board, other_board)
        I = c4_board.get_nn_input(board, other_board, with_side_to_play=True)
        O = nn.forward(I)
        EO = node.get_D(D_len=nn.L[-1])
        nn.backward(EO, d_loss_f)
        loss += loss_f(O, EO)
        loss_count += 1
        if nn.batch_count < batch_sizes[batch_index]:
            continue
        nn.update(lr, mom)
        batch_index += 1
    return

loss_mean_scale: float = 10
def distribution_loss(O, EO) -> float:
    assert len(O) == len(EO)
    return loss_mean_scale * (mysearch.get_D_mean(O) - mysearch.get_D_mean(EO)) ** 2 \
        + sum((o - eo) ** 2 for o, eo in zip(O, EO)) / len(O)

def d_distribution_loss(O, EO) -> [float]:
    assert len(O) == len(EO)
    mean_diff = (mysearch.get_D_mean(O) - mysearch.get_D_mean(EO))
    return [loss_mean_scale * 2 * mean_diff * (i + 0.5) / len(O) \
            + 2 * (o - eo) / len(O) \
            for i, (o, eo) in enumerate(zip(O, EO))]


# return wether @nn beats @old_nn after @game_pairs,
# both use same @search_params
def validation_match(nn: brain.NN, old_nn: brain.NN, game_pairs: int,
                     search_params: mysearch.Params) -> bool:
    score_delta = 0
    for i in range(1, game_pairs+1):
        print(f"\ngame pair {i}/{game_pairs}")
        print("new=X vs old=O:")
        score_delta += validation_match_game(nn, old_nn, search_params,
                name="new", other_name="old", game_nr=i * 2 - 1, game_pairs=game_pairs,
                score_delta=score_delta)
        print("new=O vs old=X:")
        score_delta -= validation_match_game(old_nn, nn, search_params,
                name="old", other_name="new", game_nr=i * 2, game_pairs=game_pairs,
                score_delta=score_delta)
        print("score delta:", score_delta)
    return score_delta > 0

# return 1 if nn (X) wins, -1 if other_nn (O) wins, 0 if draw
def validation_match_game(nn, other_nn, search_params, *,
                          name: str, other_name: str, game_nr: int, game_pairs: int,
                          score_delta: float) -> int:
    board = other_board = 0
    search_params.heuristic_evaluation = nn_evaluation
    TT: {(int, int): mysearch.Node} = {}
    other_TT: {(int, int): mysearch.Node} = {}
    regret = 0
    other_regret = 0
    random_start_plys = 5
    for nr in range(42):
        print(f"validation game {game_nr}/{2 * game_pairs}: " \
              + f"('{name}'={'O' if nr % 2 else 'X'} vs " \
              + f"'{other_name}'={'X' if nr % 2 else 'O'}): " \
              + f"score delta: {score_delta}")
        if nr >= 2:
            print("current player's regret:", round(regret, 4),
                  "mean =", round(regret / (nr//2), 4), end=", ")
            print("other player's regret:", round(other_regret, 4),
                  "mean =", round(other_regret / (nr//2), 4))
        c4_board.print_board(board, other_board)
        search_params.evaluation_params = nn
        result, stats = mysearch.search(board, other_board, search_params, TT)
        mysearch.search_info(search_params, TT, result, None)
        if result.root.solution == mysearch.Solution.HEURISTIC and nr < 2 * random_start_plys:
            # make weighted random moves in the beginning
            # to not always play the same game in the match
            for x in result.root_child_results:
                if x[1]:
                    x[1].D = x[1].get_D(search_params.D_len)
            candidates = [(m, n) for m, n in result.root_child_results if n is not None and n.solution != mysearch.Solution.WIN]
            move_index = mysearch.thompson_select([n for _, n in candidates])
            move = candidates[move_index][0]
        else:
            move = result.best_move
        assert c4_board.is_valid_move(board, other_board, move)
        move_regret = result.calc_regret_of_move(move)
        print("make move:", move + 1, "move regret:", round(move_regret, 4))
        new_board = c4_board.make_move(board, other_board, move)
        regret += move_regret
        if c4_board.is_win(new_board):
            return 1 if nr % 2 == 0 else -1
        board, other_board = other_board, new_board
        TT, other_TT= other_TT, TT
        nn, other_nn = other_nn, nn
        regret, other_regret = other_regret, regret
        name, other_name = other_name, name
    return 0

def generate_selfplay_data(search_params: mysearch.Params, \
                           n_games: int, min_nodes: int, data_epoch: int) \
                           -> [((int, int), mysearch.Node)]:
    original_initial_energy = search_params.initial_energy
    board = other_board = 0
    TT: {(int, int): mysearch.Node} = {}
    game_count = 0
    data_size = 0
    while game_count < n_games:
        print(f"data epoch {data_epoch}, game {game_count+1}/{n_games}, data-size >= {data_size}")
        c4_board.print_board(board, other_board)
        result, stats = mysearch.search(board, other_board, search_params, TT)
        mysearch.search_info(search_params, TT, result, None)
        search_params.initial_energy = min(search_params.expansions_threshold, \
                                           stats.each_epoch[-1].energy // 2)
        candidates: [(int, Node)] = [x for x in result.root_child_results \
                                if x[1] and x[1].solution == mysearch.Solution.HEURISTIC]
        if not candidates:
            board = other_board = 0
            game_count += 1
            TT = {(b, ob): node for (b, ob), node in TT.items() if node.nodes > min_nodes}
            data_size = len(TT)
            continue
        move_index = mysearch.thompson_select([n for _, n in candidates])
        move = candidates[move_index][0]
        assert c4_board.is_valid_move(board, other_board, move)
        print("make move:", move + 1)
        new_board = c4_board.make_move(board, other_board, move)
        board, other_board = other_board, new_board
    search_params.initial_energy = original_initial_energy
    return [(b, ob, node) for (b, ob), node in TT.items() if node.nodes > min_nodes]

def nn_evaluation(board: int, other_board: int, nn: brain.NN) -> [float]:
    nn_in = c4_board.get_nn_input(board, other_board, with_side_to_play=True)
    nn_out = nn.forward(nn_in)
    return processed_nn_output(nn_out)

def processed_nn_output(nn_out: [float]) -> [float]:
    offset = 0.5 / len(nn_out)
    mini = min(nn_out)
    s = sum(nn_out) + (offset - mini) * len(nn_out)
    if s == 0:
        return [1 / len(nn_out) for _ in range(len(nn_out))]
    return [(v + offset - mini) / s for v in nn_out]

if __name__ == "__main__":
    main()
