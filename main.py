import c4_board
import mysearch
import brain
import training

from typing import Any, Callable

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
def game(playerX: Callable, playerO: Callable,
         *, playerX_args: Any = None, playerO_args: None =None, show=True) -> str:
    boardX = boardO = 0
    result = c4_board.DRAW
    board, other_board= boardX, boardO
    player, other_player= playerX, playerO
    args, other_args = playerX_args, playerO_args
    for i in range(42):
        if show:
            c4_board.print_board(board, other_board)
        move = player(board, other_board, args)
        if not c4_board.is_valid_move(board, other_board, move):
            result = c4_board.WIN_X if i % 2 == 1 else c4_board.WIN_O
            break
        board, other_board = other_board, c4_board.make_move(board, other_board, move)
        player, other_player = other_player, player
        args, other_args = other_args, args
        if c4_board.is_win(other_board):
            result = c4_board.WIN_X if i % 2 == 0 else c4_board.WIN_O
            break

    if show:
        c4_board.print_board(board, other_board)
        print(f"result: {result}")
    return result

def main():
    print("main.py - Connect-Four")
    nn = brain.NN(filepath=training.get_latest_net_filepath())

    params = mysearch.Params();
    params.max_epochs = 9
    params.D_len = nn.L[-1]
    params.evaluation_params = nn
    params.heuristic_evaluation = training.nn_evaluation
    mysearch.test()
    game(mysearch.player, get_human_move, playerX_args=params)
    return

if __name__ == "__main__":
    main()
