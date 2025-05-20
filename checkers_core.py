# checkers_core.py
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from .board import Board
from .utils import CheckersError

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('checkers_core.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_to_json(message: str, level: str, extra_data: Optional[dict] = None):
    try:
        log_entry = {
            "timestamp": np.datetime64('now').astype(str),
            "level": level,
            "message": message,
            **(extra_data or {})
        }
        # Convert numpy arrays and numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        log_entry = {k: convert_numpy(v) for k, v in log_entry.items()}
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "checkers_core_log.json", "a", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        logger.error(f"Error logging to JSON: {e}")

def get_piece_moves(board: Board, row: int, col: int, player: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Returns possible moves for a piece at (row, col) for the given player.
    Format: {(to_row, to_col): [skipped_positions]}
    """
    try:
        moves = {}
        if not (0 <= row < board.board_size and 0 <= col < board.board_size):
            log_to_json(
                f"Invalid position ({row}, {col})",
                level="ERROR",
                extra_data={"row": row, "col": col, "board_size": board.board_size}
            )
            return moves
        piece = board.board[row, col]
        if piece == 0 or (player == 1 and piece < 0) or (player == -1 and piece > 0):
            log_to_json(
                f"No valid piece at ({row}, {col}) for player {player}",
                level="DEBUG",
                extra_data={"piece": piece, "player": player}
            )
            return moves
        is_king = abs(piece) == 2
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else (
            [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        )
        for dr, dc in directions:
            # Simple move
            r, c = row + dr, col + dc
            if 0 <= r < board.board_size and 0 <= c < board.board_size and board.board[r, c] == 0:
                moves[(r, c)] = []
            # Jump move
            r, c = row + 2 * dr, col + 2 * dc
            mid_r, mid_c = row + dr, col + dc
            if (0 <= r < board.board_size and 0 <= c < board.board_size and
                board.board[r, c] == 0 and
                board.board[mid_r, mid_c] != 0 and
                (player == 1 and board.board[mid_r, mid_c] < 0) or
                (player == -1 and board.board[mid_r, mid_c] > 0)):
                moves[(r, c)] = [(mid_r, mid_c)]
        log_to_json(
            f"Calculated moves for piece at ({row}, {col}): {moves}",
            level="DEBUG",
            extra_data={"player": player, "piece": piece, "moves": moves}
        )
        return moves
    except Exception as e:
        log_to_json(
            f"Error in get_piece_moves: {str(e)}",
            level="ERROR",
            extra_data={"row": row, "col": col, "player": player}
        )
        return {}

def make_move(board: Board, move: Tuple[int, int, int, int], player_number: int) -> Tuple[Optional[Board], bool, bool]:
    """
    Executes a move and returns (new_board, is_promotion, has_more_jumps).
    """
    try:
        from_row, from_col, to_row, to_col = move
        if not (0 <= from_row < board.board_size and 0 <= from_col < board.board_size and
                0 <= to_row < board.board_size and 0 <= to_col < board.board_size):
            log_to_json(
                f"Invalid move coordinates: {move}",
                level="ERROR",
                extra_data={"move": move, "board_size": board.board_size}
            )
            return None, False, False
        piece = board.board[from_row, from_col]
        player = 1 if player_number == 1 else -1
        if piece == 0 or (player == 1 and piece < 0) or (player == -1 and piece > 0):
            log_to_json(
                f"Invalid piece at ({from_row}, {from_col}) for player {player}",
                level="ERROR",
                extra_data={"piece": piece, "player": player}
            )
            return None, False, False
        valid_moves = get_piece_moves(board, from_row, from_col, player)
        if (to_row, to_col) not in valid_moves:
            log_to_json(
                f"Move {move} not valid",
                level="ERROR",
                extra_data={"move": move, "valid_moves": valid_moves}
            )
            return None, False, False
        new_board = board.copy()
        new_board.board[to_row, to_col] = piece
        new_board.board[from_row, from_col] = 0
        is_jump = abs(to_row - from_row) == 2
        skipped = valid_moves[(to_row, to_col)]
        for skip_r, skip_c in skipped:
            new_board.board[skip_r, skip_c] = 0
            if player == 1:
                new_board.player_2_left -= 1
            else:
                new_board.player_1_left -= 1
        is_promotion = False
        if player == 1 and to_row == 0 and abs(piece) == 1:
            new_board.board[to_row, to_col] = 2
            is_promotion = True
        elif player == -1 and to_row == board.board_size - 1 and abs(piece) == 1:
            new_board.board[to_row, to_col] = -2
            is_promotion = True
        has_more_jumps = False
        if is_jump and not is_promotion:
            additional_jumps = get_piece_moves(new_board, to_row, to_col, player)
            has_more_jumps = any(abs(to_r - to_row) == 2 for to_r, to_c in additional_jumps.keys())
        log_to_json(
            f"Move {move} executed: is_jump={is_jump}, is_promotion={is_promotion}, has_more_jumps={has_more_jumps}",
            level="INFO",
            extra_data={"move": move, "player": player}
        )
        return new_board, is_promotion, has_more_jumps
    except Exception as e:
        log_to_json(
            f"Error in make_move: {str(e)}",
            level="ERROR",
            extra_data={"move": move, "player_number": player_number}
        )
        return None, False, False

def get_legal_moves(board: Board, player: int) -> List[Tuple[int, int, int, int]]:
    """
    Returns all legal moves for the player in 4-tuple format.
    """
    try:
        moves = []
        for row in range(board.board_size):
            for col in range(board.board_size):
                if board.board[row, col] * player > 0:
                    piece_moves = get_piece_moves(board, row, col, player)
                    for (to_row, to_col), _ in piece_moves.items():
                        moves.append((row, col, to_row, to_col))
        log_to_json(
            f"Legal moves for player {player}: {moves}",
            level="DEBUG",
            extra_data={"player": player, "move_count": len(moves)}
        )
        return moves
    except Exception as e:
        log_to_json(
            f"Error in get_legal_moves: {str(e)}",
            level="ERROR",
            extra_data={"player": player}
        )
        return []