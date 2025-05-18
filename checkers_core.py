#checkers_core.py
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Optional
from .board import Board
from .config import load_config


def is_valid_position(row: int, col: int, board_size: int) -> bool:
    return 0 <= row < board_size and 0 <= col < board_size


def can_jump(board: Board, from_row: int, from_col: int, mid_row: int, mid_col: int, piece: int) -> bool:
    opponent_piece = board.board[mid_row, mid_col]
    if opponent_piece == 0 or opponent_piece * piece > 0:
        return False
    to_row, to_col = from_row + 2 * (mid_row - from_row), from_col + 2 * (mid_col - from_col)
    if not is_valid_position(to_row, to_col, board.board_size):
        return False
    if board.board[to_row, to_col] != 0:
        return False
    return True


def get_piece_moves(board: Board, row: int, col: int, player: Optional[int] = None) -> Dict[Tuple[int, int], set[Tuple[int, int]]]:
    moves = {}
    piece = board.board[row, col]
    if piece == 0 or (player is not None and piece * player <= 0):
        return moves

    # تعریف جهت‌های ممکن
    all_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # بالا-چپ، بالا-راست، پایین-چپ، پایین-راست
    if abs(piece) == 1:  # مهره عادی
        if piece > 0:  # بازیکن ۱ (سفید، حرکت به بالا)
            directions = [(-1, -1), (-1, 1)]
        else:  # بازیکن ۲ (سیاه، حرکت به پایین)
            directions = [(1, -1), (1, 1)]
    else:  # مهره شاه
        directions = all_directions

    # منطق محاسبه حرکات
    for direction in directions:
        new_row, new_col = row + direction[0], col + direction[1]
        if is_valid_position(new_row, new_col, board.board_size):
            if board.board[new_row, new_col] == 0:
                moves[(new_row, new_col)] = set()  # حرکات ساده
            elif can_jump(board, row, col, new_row, new_col, piece):
                jump_row, jump_col = row + 2 * direction[0], col + 2 * direction[1]
                skipped = {(new_row, new_col)}
                moves[(jump_row, jump_col)] = skipped
    return moves


def make_move(board: Board, move: Tuple[int, int, int, int], player_number: int) -> Tuple[Optional[Board], bool, bool]:
    from_row, from_col, to_row, to_col = move
    if not is_valid_position(from_row, from_col, board.board_size) or not is_valid_position(to_row, to_col, board.board_size):
        log_to_json(
            f"Invalid move coordinates: {move}",
            level="ERROR",
            extra_data={"move": move}
        )
        return None, False, False
    new_board = Board(board.settings)
    new_board.board = board.board.copy()
    new_board.player_1_left = board.player_1_left
    new_board.player_2_left = board.player_2_left
    piece = new_board.board[from_row, from_col]
    player = 1 if player_number == 1 else -1

    if piece == 0 or piece * player < 0:
        log_to_json(
            f"Invalid piece for move {move}: piece={piece}, player={player}",
            level="ERROR",
            extra_data={"move": move, "board": board.board.tolist()}
        )
        return None, False, False

    moves = get_piece_moves(new_board, from_row, from_col)
    log_to_json(
        f"Available moves for piece at ({from_row}, {from_col}): {moves}",
        level="DEBUG",
        extra_data={"move": move, "board": board.board.tolist()}
    )
    if (to_row, to_col) not in moves:
        log_to_json(
            f"Move {move} not in valid moves: {moves}",
            level="ERROR",
            extra_data={"move": move, "board": board.board.tolist()}
        )
        return None, False, False

    new_board.board[to_row, to_col] = piece
    new_board.board[from_row, from_col] = 0
    skipped = moves[(to_row, to_col)]
    new_board.remove(skipped)

    is_jump = len(skipped) > 0
    is_promotion = False
    if player == 1 and to_row == new_board.board_size - 1 and abs(piece) == 1:
        new_board.board[to_row, to_col] = 2
        is_promotion = True
    elif player == -1 and to_row == 0 and abs(piece) == 1:
        new_board.board[to_row, to_col] = -2
        is_promotion = True

    has_more_jumps = False
    if is_jump and not is_promotion:
        jumps = get_piece_moves(new_board, to_row, to_col)
        has_more_jumps = any(skipped for _, skipped in jumps.items())

    return new_board, is_promotion, has_more_jumps


def log_to_json(message: str, level: str = "INFO", extra_data: Optional[Dict] = None):
    """Logs a message to a JSON file based on the logging level."""
    config = load_config()
    logging_level = config.get("logging_level", "ERROR")
    level_priority = {"INFO": 1, "WARNING": 2, "ERROR": 3}

    if level_priority.get(level, 0) < level_priority.get(logging_level, 3):
        return

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "extra_data": extra_data or {}
    }

    log_file = os.path.join(os.path.dirname(__file__), "checkers_errors.json")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        log_to_json(f"Failed to log to JSON: {str(e)}", "ERROR")