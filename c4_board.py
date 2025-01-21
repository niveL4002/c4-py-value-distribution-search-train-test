WIN_X = "X wins"
WIN_O = "O wins"
DRAW = "draw"

def print_board(board: int, other_board: int):
    print(" ", end="")
    for m in range(7):
        print(str(m + 1) if is_valid_move(board, other_board, m) else "-", end=" ")
    print()
    move_count = get_move_count(board, other_board)
    board_sign = "O" if (move_count % 2) else "X"
    other_board_sign = "X" if (move_count % 2) else "O"
    for i in range(5, -1, -1):
        mask = 1 << i
        print("|", end="")
        for _ in range(7):
            if board & mask:
                print(board_sign, end="|")
            elif other_board & mask:
                print(other_board_sign, end="|")
            else:
                print("_", end="|")
            mask <<= 7
        if (i == 3):
            print(f"  move_count: {move_count}")
        elif (i == 2):
            print("  ", end="")
            if is_win(board) or is_win(other_board):
                print(WIN_X if move_count % 2 == 1 else WIN_O)
            elif move_count == 42:
                print(DRAW)
            else:
                print(f"{board_sign}'s turn")
        else:
            print()
    return

def is_win(board: int) -> bool:
    # vertical
    tmp = board & (board >> 1)
    if tmp & (tmp >> 2):
        return True
    # horizontal
    tmp = board & (board >> 7)
    if tmp & (tmp >> 14):
        return True
    # diagonal \
    tmp = board & (board >> 6)
    if tmp & (tmp >> 12):
        return True
    # diagonal /
    tmp = board & (board >> 8)
    if tmp & (tmp >> 16):
        return True
    # no win
    return False

def is_valid_move(board: int, other_board: int, col: int) -> bool:
    return (0 <= col < 7) and not ( (1 << col*7+5) & (board | other_board) )

def make_move(board: int, other_board: int, col: int) ->int:
    # returns updated board with move made at col
    return board | (((63 << 7*col) & (board | other_board)) + (1 << 7*col))

def get_move_count(board1: int, board2: int) -> int:
    return (board1 | board2).bit_count()

def get_mirrored(board: int, other_board) -> (int, int):
    b = o = 0
    for _ in range(7):
        b <<= 7
        o <<= 7
        b |= 63 & board
        o |= 63 & other_board
        board >>= 7
        other_board >>= 7
    return (b, o)

nn_input_masks = [1 << (i // 7 + 7 * (i % 7)) for i in range(42)]
def get_nn_input(board: int, other_board: int,
                 with_side_to_play: bool = False) -> [float]:
    res = [bool(mask & board) - bool(mask & other_board) for mask in nn_input_masks]
    if with_side_to_play:
         res.append(get_move_count(board, other_board) % 2)
    return res

def print_nn_input(nn_input: [float]):
    for row in range(5, -1, -1):
        for col in range(7):
            print(f"{nn_input[row * 7 + col]:2}", end=" ")
        if row == 0 and len(nn_input) == 43:
            print("  : odd_even =", nn_input[-1])
        print()
    return







