from typing import Dict, Optional
from pyfile.checkers_core import get_piece_moves, make_move
from pyfile.config import load_config, load_ai_config, log_to_json, DEFAULT_AI_PARAMS
from pyfile.utils import CheckersError

class RewardCalculator:
    """Calculates rewards for the checkers game based on game state.

    Attributes:
        game: The Game object.
        ai_id (str): Identifier for the AI ('ai_1' or 'ai_2'), or None.
        board_size (int): Size of the board (e.g., 8 for 8x8).
        weights (dict): Weights for different reward components.
        end_game_rewards (dict): Rewards for end-game outcomes.
    """
    def __init__(self, game, ai_id: Optional[str] = None):
        self.game = game
        self.ai_id = ai_id
        config = load_config()
        self.board_size = config.get("board_size", 8)
        self.weights, self.end_game_rewards = self._load_config()

    def _load_config(self) -> tuple[Dict[str, float], Dict[str, float]]:
        """Loads reward weights and end-game rewards from configuration."""
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        try:
            ai_config = load_ai_config()
            player_config = ai_config["ai_configs"].get(player_key, {})
            params = player_config.get("params", DEFAULT_AI_PARAMS)
            weights = params.get("reward_weights", DEFAULT_AI_PARAMS["reward_weights"])
            end_game_rewards = params.get("end_game_rewards", DEFAULT_AI_PARAMS["end_game_rewards"])
            if not isinstance(weights, dict) or not isinstance(end_game_rewards, dict):
                raise ValueError("Reward weights or end-game rewards are not dictionaries")
            log_to_json(f"Loaded config for {player_key} for AI {self.ai_id}", level="INFO")
            return weights, end_game_rewards
        except Exception as e:
            log_to_json(f"Error loading config: {str(e)}, using default weights", level="ERROR")
            return DEFAULT_AI_PARAMS["reward_weights"], DEFAULT_AI_PARAMS["end_game_rewards"]

    def get_reward(self, player_number: Optional[int] = None) -> float:
        """Calculates the total reward for the current game state."""
        if player_number is None and self.ai_id:
            player_number = 1 if self.ai_id == "ai_1" else 2
        if player_number not in [1, 2]:
            log_to_json(
                f"Invalid player_number {player_number}, returning 0 reward",
                level="WARNING",
                extra_data={"player_number": player_number}
            )
            return 0.0
        if self.game.game_over:
            return self._end_game_reward(player_number)
        return self._in_game_reward(player_number)

    def _end_game_reward(self, player_number: int) -> float:
        """Calculates the reward for end-game states."""
        if self.game.winner is not None:
            is_player = (player_number == 2)  # player_number=2 is black
            winner_is_player = self.game.winner  # True for black, False for white
            if is_player == winner_is_player:
                return (self.end_game_rewards['win_no_timeout']
                        if not getattr(self.game, 'time_up', False)
                        else self.end_game_rewards['win_timeout'])
            return self.end_game_rewards['loss']
        return self.end_game_rewards['draw']

    def _in_game_reward(self, player_number: int) -> float:
        """Calculates the reward for in-game states."""
        reward = 0.0
        reward += self._piece_difference_reward(player_number) * self.weights['piece_difference']
        reward += self._king_bonus_reward(player_number) * self.weights['king_bonus']
        reward += self._position_bonus_reward(player_number) * self.weights['position_bonus']
        reward += self._capture_bonus_reward(player_number) * self.weights['capture_bonus']
        reward += self._multi_jump_bonus_reward(player_number) * self.weights['multi_jump_bonus']
        reward += self._king_capture_bonus_reward(player_number) * self.weights['king_capture_bonus']
        reward += self._mobility_bonus_reward(player_number) * self.weights['mobility_bonus']
        reward += self._safety_penalty_reward(player_number) * self.weights['safety_penalty']
        return reward

    def _piece_difference_reward(self, player_number: int) -> float:
        """Calculates reward based on piece count difference."""
        player_pieces = sum(
            1 for row in range(self.board_size) for col in range(self.board_size)
            if (self.game.board.board[row, col] > 0 and player_number == 1) or
               (self.game.board.board[row, col] < 0 and player_number == 2)
        )
        opponent_pieces = sum(
            1 for row in range(self.board_size) for col in range(self.board_size)
            if (self.game.board.board[row, col] > 0 and player_number == 2) or
               (self.game.board.board[row, col] < 0 and player_number == 1)
        )
        return player_pieces - opponent_pieces

    def _king_bonus_reward(self, player_number: int) -> float:
        """Calculates reward based on king count."""
        king_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and abs(piece) == 2:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    king_bonus += 1 if is_player else -1
        return king_bonus

    def _position_bonus_reward(self, player_number: int) -> float:
        """Calculates reward based on piece positions (closer to king row)."""
        position_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    distance_to_king = row if player_number == 2 else self.board_size - 1 - row
                    position_bonus += (self.board_size - distance_to_king) if is_player else -(self.board_size - distance_to_king)
        return position_bonus

    def _capture_bonus_reward(self, player_number: int) -> float:
        """Calculates reward for possible captures."""
        capture_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0:
                    moves = get_piece_moves(self.game.board, row, col, player)
                    if any(len(skipped) > 0 for _, skipped in moves.items()):
                        capture_bonus += 1
        return capture_bonus

    def _multi_jump_bonus_reward(self, player_number: int) -> float:
        """Calculates reward for possible multi-jumps."""
        multi_jump_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0:
                    multi_jump_count = self._count_multi_jumps(row, col, set(), player)
                    multi_jump_bonus += multi_jump_count
        return multi_jump_bonus

    def _count_multi_jumps(self, row: int, col: int, visited: set, player: int) -> int:
        """Counts possible multi-jumps for a piece using a temporary board state."""
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        temp_board = self.game.board.copy()
        moves = get_piece_moves(temp_board, row, col, player)
        multi_jump_count = 0

        for (to_row, to_col), skipped in moves.items():
            if skipped:
                move = (row, col, to_row, to_col)
                try:
                    result = make_move(temp_board, move, player)
                    new_board, _, _ = result if result else (None, False, False)
                    if new_board:
                        multi_jump_count += 1
                        multi_jump_count += self._count_multi_jumps(to_row, to_col, visited.copy(), player)
                    temp_board = self.game.board.copy()
                except CheckersError:
                    continue
        return multi_jump_count

    def _king_capture_bonus_reward(self, player_number: int) -> float:
        """Calculates reward for captures leading to king promotion."""
        king_capture_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0 and abs(piece) != 2:
                    moves = get_piece_moves(self.game.board, row, col, player)
                    for (to_row, to_col), skipped in moves.items():
                        if skipped:
                            end_row = to_row
                            if (player_number == 1 and end_row == self.board_size - 1) or (player_number == 2 and end_row == 0):
                                king_capture_bonus += 1
        return king_capture_bonus

    def _mobility_bonus_reward(self, player_number: int) -> float:
        """Calculates reward for piece mobility."""
        mobility_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = piece * player > 0
                    moves = get_piece_moves(self.game.board, row, col, player)
                    mobility_bonus += len(moves) if is_player else -len(moves)
        return mobility_bonus

    def _safety_penalty_reward(self, player_number: int) -> float:
        """Calculates penalty for pieces at risk of being captured."""
        safety_penalty = 0
        player = 1 if player_number == 1 else -1
        opponent = -player
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0:
                    for r in range(self.board_size):
                        for c in range(self.board_size):
                            opp_piece = self.game.board.board[r, c]
                            if opp_piece != 0 and opp_piece * opponent > 0:
                                moves = get_piece_moves(self.game.board, r, c, opponent)
                                for (to_row, to_col), skipped in moves.items():
                                    if skipped:
                                        mid_row, mid_col = (r + to_row) // 2, (c + to_col) // 2
                                        if (mid_row, mid_col) == (row, col):
                                            safety_penalty -= 1
        return safety_penalty