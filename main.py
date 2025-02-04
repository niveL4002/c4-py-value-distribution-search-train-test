import c4_board
import mysearch
import brain
import training

from typing import Any, Callable
import sys

def get_human_move(board: int, other_board: int, args: Any = None) -> int:
    while True:
        try:
            col = int(input("Enter valid move: ")) - 1
        except ValueError:
            continue
        if c4_board.is_valid_move(board, other_board, col):
            break
    return col

# return result
def game(playerX: Callable, playerO: Callable, X_name: str = None, O_name: str = None,
         *, playerX_args: Any = None, playerO_args: Any = None, show=True) -> str:
    boardX = boardO = 0
    result = c4_board.DRAW
    winner_name = None
    board, other_board= boardX, boardO
    player, other_player= playerX, playerO
    name, other_name = X_name, O_name
    args, other_args = playerX_args, playerO_args
    for i in range(42):
        if show:
            if name is not None:
                print(name + "/" + ("X" if i % 2 == 0 else "O") + " to play ...")
            c4_board.print_board(board, other_board)
        move = player(board, other_board, args)
        if not c4_board.is_valid_move(board, other_board, move):
            result = c4_board.WIN_X if i % 2 == 1 else c4_board.WIN_O
            break
        board, other_board = other_board, c4_board.make_move(board, other_board, move)
        player, other_player = other_player, player
        name, other_name = other_name, name
        args, other_args = other_args, args
        if c4_board.is_win(other_board):
            winner_name = other_name
            result = c4_board.WIN_X if i % 2 == 0 else c4_board.WIN_O
            break

    if show:
        c4_board.print_board(board, other_board)
        print(f"result: {'' if winner_name is None else winner_name + '/'}{result}")
    return result

# returns (get_move_function(), argument_to_get_move_function(), name_of_player)
def get_player_from_arg(arg: str) -> (Callable, Any, str):
    if arg == "h":
        return (get_human_move, None, "human")
    # get computer player
    if arg == "c":
        nr = training.get_latest_net_index()
    else:
        try:
            nr = int(arg)
        except ValueError:
            print("invalid input")
            return
    nn = brain.NN(filepath=training.get_net_filepath(nr))
    params = mysearch.Params();
    params.max_epochs = 9
    params.D_len = nn.L[-1]
    params.evaluation_params = nn
    params.heuristic_evaluation = training.nn_evaluation
    return (mysearch.player, params, "computerNN" + str(nr))

def main():
    print("main - Connect-Four")
    argc: int = len(sys.argv)
    if argc > 3:
        print("invalid argument count")
        return
    print(sys.argv)
    if argc == 2 and sys.argv[1] == "test":
        mysearch.test()
        return
    playerX, X_args, X_name = get_player_from_arg(sys.argv[1] if argc > 1 else "h")
    playerO, O_args, O_name = get_player_from_arg(sys.argv[2] if argc > 2 else "c")
    game(playerX, playerO, X_name, O_name, playerX_args=X_args, playerO_args=O_args)
    return

if __name__ == "__main__":
    main()
