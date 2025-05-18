# checkers_game.py
import logging
import os
import numpy as np
from typing import Tuple
from .board import Board
from .checkers_core import get_piece_moves, make_move, log_to_json
from .config import load_config
from .settings import GameSettings
from .utils import CheckersError


# Load configuration
config = load_config()
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkers_game.log')
logging.basicConfig(
    level=getattr(logging, config.get("logging_level", "ERROR")),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)


class CheckersGame:
    """Manages the checkers game logic.

    Attributes:
        board (Board): Current board state.
        current_player (int): 1 for player 1, -1 for player 2.
        move_count (int): Number of moves made.
    """

    def __init__(self, settings=None):
        self.config = load_config()
        # تغییر: استفاده از settings ورودی یا ایجاد نمونه جدید GameSettings
        self.settings = settings if settings else GameSettings()
        self.board = Board(self.settings)  # تغییر: پاس دادن self.settings به Board
        self.current_player = 1
        self.move_count = 0

    def reset(self):
        """Resets the game to the initial state."""
        self.board = Board(self.settings)  # تغییر: استفاده از self.settings
        self.current_player = 1
        self.move_count = 0
        log_to_json("Game reset", level="INFO")

    def copy(self):
        """Returns a deep copy of the game state."""
        new_game = CheckersGame()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def set_state(self, state):
        """Sets the board state from an 8x8 array.

        Args:
            state (np.ndarray): State array with shape (8, 8).
        """
        if state.shape != (self.board.board_size, self.board.board_size):
            log_to_json(
                f"Invalid state shape: {state.shape}",
                level="ERROR",
                extra_data={"expected_shape": (self.board.board_size, self.board.board_size)}
            )
            raise CheckersError("Invalid state shape")
        self.board.board = state.copy()
        # Update piece counts
        self.board.player_1_left = np.sum(self.board.board > 0)
        self.board.player_2_left = np.sum(self.board.board < 0)
        log_to_json("State set successfully", level="DEBUG")

    def get_state(self):
        """Returns the current state as an 8x8 array.

        Returns:
            np.ndarray: State array.
        """
        return self.board.board.copy()

    @staticmethod
    def get_action_size():
        """Returns the size of the action space.

        Returns:
            int: Number of possible actions (32 * 32).
        """
        return 32 * 32

    def get_legal_moves(self):
        """Returns all legal moves for the current player.

        Returns:
            list: List of moves as (from_row, from_col, to_row, to_col).
        """
        legal_moves = []
        jumps_exist = False
        # Check jumps
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * self.current_player > 0:
                    moves = get_piece_moves(self.board, row, col)
                    jumps = [(to_pos, skipped) for to_pos, skipped in moves.items() if skipped]
                    if jumps:
                        jumps_exist = True
                        legal_moves.extend([(row, col, to_row, to_col) for (to_row, to_col), _ in jumps])
        if jumps_exist:
            log_to_json(
                f"Legal jumps for player {self.current_player}: {legal_moves}",
                level="DEBUG"
            )
            return legal_moves
        # Check simple moves
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * self.current_player > 0:
                    moves = get_piece_moves(self.board, row, col)
                    legal_moves.extend(
                        [(row, col, to_row, to_col) for (to_row, to_col), skipped in moves.items() if not skipped])
        log_to_json(
            f"Legal moves for player {self.current_player}: {legal_moves}",
            level="DEBUG"
        )
        return legal_moves

    def make_move(self, move: Tuple[int, int, int, int]) -> 'CheckersGame':
        """Executes a move and updates the game state.

        Args:
            move (tuple): Move as (from_row, from_col, to_row, to_col).

        Returns:
            CheckersGame: Updated game instance.
        """
        try:
            from_row, from_col, to_row, to_col = move
            new_board = make_move(self.board, move)
            if new_board is None:
                raise CheckersError(f"Invalid move: {move}")

            self.board = new_board
            self.move_count += 1
            is_jump = abs(to_row - from_row) == 2

            # Check for promotion
            is_promotion = (self.current_player == 1 and to_row == self.board.board_size - 1 and self.board.board[
                to_row, to_col] == 2) or \
                           (self.current_player == -1 and to_row == 0 and self.board.board[to_row, to_col] == -2)

            self.current_player = -self.current_player
            if is_jump and not is_promotion:
                jumps = get_piece_moves(self.board, to_row, to_col)
                jumps = [(to_pos, skipped) for to_pos, skipped in jumps.items() if skipped]
                if jumps:
                    self.current_player = -self.current_player
                    self.move_count -= 1
                    log_to_json(
                        f"Additional jumps available at ({to_row}, {to_col}): {jumps}",
                        level="INFO",
                        extra_data={"move": move}
                    )

            log_to_json(
                f"Made move: {move}",
                level="INFO",
                extra_data={"is_jump": is_jump, "is_promotion": is_promotion, "current_player": self.current_player}
            )
            return self
        except CheckersError as e:
            log_to_json(
                f"CheckersError in make_move: {str(e)}",
                level="ERROR",
                extra_data={"move": move}
            )
            raise
        except Exception as e:
            log_to_json(
                f"Unexpected error in make_move: {str(e)}",
                level="ERROR",
                extra_data={"move": move, "board": self.board.board.tolist()}
            )
            raise

    def is_terminal(self):
        """Checks if the game is in a terminal state.

        Returns:
            bool: True if the game is over.
        """
        return not self.get_legal_moves() or self.move_count >= 100

    def get_reward(self):
        """Returns the reward for the current state.

        Returns:
            float: Reward value.
        """
        if not self.is_terminal():
            return 0
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return -1 if self.current_player == 1 else 1
        return 0

    def get_outcome(self):
        """Returns the game outcome.

        Returns:
            float: Outcome value.
        """
        return self.get_reward()