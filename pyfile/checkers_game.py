import logging
import os
import numpy as np
from typing import Tuple
from modules.board import Board
from modules.checkers_core import get_piece_moves, make_move
from .config import ConfigManager, log_to_json
from .settings import GameSettings
from .utils import CheckersError

# نمونه جهانی ConfigManager
_config_manager = ConfigManager()

# تنظیم لاگینگ با استفاده از ConfigManager
config = _config_manager.load_config()
project_root = _config_manager.get_project_root()
log_file = os.path.join(project_root, config.get("log_file", "logs/checkers_game.log"))
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.get("logging_level")),
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
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

    def __init__(self, settings: GameSettings | None = None):
        self.config = _config_manager.load_config()
        self.settings = settings if settings else GameSettings()
        self.board = Board(self.settings)
        self.current_player = 1
        self.move_count = 0
        log_to_json(
            "CheckersGame initialized",
            level="DEBUG",
            extra_data={
                "board_size": self.board.board_size,
                "game_mode": self.settings.game_mode
            }
        )

    def reset(self):
        """Resets the game to the initial state."""
        self.board = Board(self.settings)
        self.current_player = 1
        self.move_count = 0
        log_to_json("Game reset", level="INFO")

    def copy(self) -> 'CheckersGame':
        """Returns a deep copy of the game state."""
        new_game = CheckersGame(self.settings)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def set_state(self, state: np.ndarray):
        """Sets the board state from an array.

        Args:
            state (np.ndarray): State array with shape (board_size, board_size).
        """
        if state.shape != (self.board.board_size, self.board.board_size):
            log_to_json(
                f"Invalid state shape: {state.shape}",
                level="ERROR",
                extra_data={"expected_shape": (self.board.board_size, self.board.board_size)}
            )
            raise CheckersError("Invalid state shape")
        self.board.board = state.copy()
        self.board.player_1_left = np.sum(self.board.board > 0)
        self.board.player_2_left = np.sum(self.board.board < 0)
        log_to_json("State set successfully", level="DEBUG")

    def get_state(self) -> np.ndarray:
        """Returns the current state as an array.

        Returns:
            np.ndarray: State array.
        """
        return self.board.board.copy()

    @staticmethod
    def get_action_size() -> int:
        """Returns the size of the action space.

        Returns:
            int: Number of possible actions based on board size.
        """
        board_size = _config_manager.load_config().get("board_size", 8)
        return (board_size * board_size // 2) ** 2

    def get_legal_moves(self) -> list[Tuple[int, int, int, int]]:
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
                    moves = get_piece_moves(self.board, row, col, self.current_player)
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
                    moves = get_piece_moves(self.board, row, col, self.current_player)
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
            new_board = make_move(self.board, move, self.current_player)
            if new_board is None:
                raise CheckersError(f"Invalid move: {move}")

            self.board = new_board
            self.move_count += 1
            is_jump = abs(to_row - from_row) == 2

            # Check for promotion
            is_promotion = (self.current_player == 1 and to_row == self.board.board_size - 1 and
                           self.board.board[to_row, to_col] == 2) or \
                           (self.current_player == -1 and to_row == 0 and
                            self.board.board[to_row, to_col] == -2)

            self.current_player = -self.current_player
            if is_jump and not is_promotion:
                jumps = get_piece_moves(self.board, to_row, to_col, self.current_player)
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

    def is_terminal(self) -> bool:
        """Checks if the game is in a terminal state.

        Returns:
            bool: True if the game is over.
        """
        max_total_moves = self.config.get("max_total_moves")
        return not self.get_legal_moves() or self.move_count >= max_total_moves

    def get_reward(self) -> float:
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

    def get_outcome(self) -> float:
        """Returns the game outcome.

        Returns:
            float: Outcome value.
        """
        return self.get_reward()