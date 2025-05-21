import logging
import os
import numpy as np
from copy import deepcopy

# تنظیم لاگ‌گیری
log_dir = os.path.dirname(os.path.dirname(__file__))
log_file = os.path.join(log_dir, 'checkers_game.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)

class CheckersGame:
    def __init__(self):
        self.board = None
        self.current_player = 1
        self.move_count = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        for row in [0, 1, 2]:
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1
        for row in [5, 6, 7]:
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1
        self.current_player = 1
        self.move_count = 0
        logger.info("Game reset")

    def copy(self):
        new_game = CheckersGame.__new__(CheckersGame)
        new_game.board = deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def set_state(self, state):
        if state.shape != (4, 8, 8):
            logger.error(f"Invalid state shape: {state.shape}")
            return
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[state[0] == 1] = 1
        self.board[state[1] == 1] = 2
        self.board[state[2] == 1] = -1
        self.board[state[3] == 1] = -2
        logger.debug("State set successfully")

    def get_state(self):
        state = np.zeros((4, 8, 8), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == 2).astype(np.float32)
        state[2] = (self.board == -1).astype(np.float32)
        state[3] = (self.board == -2).astype(np.float32)
        return state

    def get_action_size(self):
        return 32 * 32

    def _idx_to_row_col(self, idx):
        if not (0 <= idx < 32):
            logger.warning(f"Invalid idx: {idx}")
            return None
        row = (idx // 4) * 2 + (1 if (idx % 4) in [0, 1] else 0)
        col_base = (idx % 4) * 2
        col = col_base + (0 if row % 2 == 0 else 1)
        if not (0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1):
            logger.warning(f"Invalid row,col from idx {idx}: ({row}, {col})")
            return None
        return row, col

    def _row_col_to_idx(self, row, col):
        if not (0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1):
            logger.warning(f"Invalid row,col: ({row}, {col})")
            return None
        idx = ((row // 2) * 4) + ((col // 2) if row % 2 == 0 else (col // 2))
        if not (0 <= idx < 32):
            logger.warning(f"Invalid idx from row,col ({row}, {col}): {idx}")
            return None
        return idx

    def _get_piece_moves(self, row, col, jumps_only=False):
        moves = []
        piece = self.board[row, col]
        if piece == 0:
            logger.debug(f"No piece at ({row}, {col})")
            return moves
        is_king = abs(piece) == 2
        player = 1 if piece > 0 else -1
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else ([(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)])
        jumps = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if not jumps_only and self.board[new_row, new_col] == 0:
                    from_idx = self._row_col_to_idx(row, col)
                    to_idx = self._row_col_to_idx(new_row, new_col)
                    if from_idx is not None and to_idx is not None and from_idx != to_idx:
                        moves.append((from_idx, to_idx))
                jump_row, jump_col = new_row + dr, new_col + dc
                if 0 <= jump_row < 8 and 0 <= jump_col < 8:
                    if self.board[new_row, new_col] * player < 0 and self.board[jump_row, jump_col] == 0:
                        from_idx = self._row_col_to_idx(row, col)
                        to_idx = self._row_col_to_idx(jump_row, jump_col)
                        if from_idx is not None and to_idx is not None and from_idx != to_idx:
                            jumps.append((from_idx, to_idx))
        moves = jumps if jumps else moves
        logger.debug(f"Moves for ({row}, {col}): {moves}")
        return moves

    def get_legal_moves(self):
        legal_moves = []
        jumps_exist = False
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1 and self.board[row, col] * self.current_player > 0:
                    jumps = self._get_piece_moves(row, col, jumps_only=True)
                    if jumps:
                        jumps_exist = True
                        legal_moves.extend(jumps)
        if jumps_exist:
            logger.debug(f"Legal jumps for player {self.current_player}: {legal_moves}")
            return legal_moves
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1 and self.board[row, col] * self.current_player > 0:
                    moves = self._get_piece_moves(row, col)
                    legal_moves.extend(moves)
        logger.debug(f"Legal moves for player {self.current_player}: {legal_moves}")
        return legal_moves

    def make_move(self, move):
        from_idx, to_idx = move
        legal_moves = self.get_legal_moves()
        if (from_idx, to_idx) not in legal_moves:
            logger.error(f"Invalid move ({from_idx}, {to_idx}), legal_moves={legal_moves}")
            raise ValueError("Invalid move")
        from_pos = self._idx_to_row_col(from_idx)
        to_pos = self._idx_to_row_col(to_idx)
        if from_pos is None or to_pos is None:
            logger.error(f"Invalid indices: from_idx={from_idx}, to_idx={to_idx}")
            raise ValueError("Invalid move")
        from_row, from_col = from_pos
        new_row, new_col = to_pos
        piece = self.board[from_row, from_col]
        self.board[from_row, from_col] = 0
        is_jump = abs(new_row - from_row) == 2
        if is_jump:
            mid_row = (from_row + new_row) // 2
            mid_col = (from_col + new_col) // 2
            self.board[mid_row, mid_col] = 0
        self.board[new_row, new_col] = piece
        if (piece == 1 and new_row == 0) or (piece == -1 and new_row == 7):
            self.board[new_row, new_col] = piece * 2
        self.current_player = -self.current_player
        self.move_count += 1
        if is_jump:
            jumps = self._get_piece_moves(new_row, new_col, jumps_only=True)
            if jumps and all((f_idx, t_idx) not in legal_moves for f_idx, t_idx in jumps):
                self.current_player = -self.current_player
                self.move_count += 1
        logger.info(f"Made move from ({from_row}, {from_col}) to ({new_row}, {new_col})")
        return self

    def is_terminal(self):
        return not self.get_legal_moves() or self.move_count >= 100

    def get_reward(self):
        if not self.is_terminal():
            return 0
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return -1 if self.current_player == 1 else 1
        return 0

    def get_outcome(self):
        return self.get_reward()