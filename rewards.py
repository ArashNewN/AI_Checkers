# rewards.py
import json
from pathlib import Path
from .checkers_core import get_piece_moves, log_to_json, make_move
from .config import load_config, load_ai_config, DEFAULT_AI_PARAMS
from .utils import CheckersError



class RewardCalculator:
    """Calculates rewards for the checkers game based on game state.

    Attributes:
        game: The Game object.
        ai_id (str): Identifier for the AI ('ai_1' or 'ai_2'), or None.
        board_size (int): Size of the board (e.g., 8 for 8x8).
        weights (dict): Weights for different reward components.
        end_game_rewards (dict): Rewards for end-game outcomes.
    """
    def __init__(self, game, ai_id=None):
        self.game = game
        self.ai_id = ai_id
        config = load_config()
        self.board_size = config.get("board_size", 8)
        self.weights, self.end_game_rewards = self._load_config()

    def _load_config(self):
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        try:
            ai_config = load_ai_config()
            params = ai_config["ai_configs"].get(player_key, {}).get("params", DEFAULT_AI_PARAMS)
            weights = params.get("reward_weights", DEFAULT_AI_PARAMS["reward_weights"])
            end_game_rewards = params.get("end_game_rewards", DEFAULT_AI_PARAMS["end_game_rewards"])
            log_to_json(f"Loaded config for {player_key} for AI {self.ai_id}", level="INFO")
            return weights, end_game_rewards
        except Exception as e:
            log_to_json(f"Error loading config: {str(e)}, using default weights", level="ERROR")
            return DEFAULT_AI_PARAMS["reward_weights"], DEFAULT_AI_PARAMS["end_game_rewards"]

    def get_reward(self, player_number=None):
        """Calculates the total reward for the current game state.

        Args:
            player_number (int, optional): 1 for white, 2 for black. If None, inferred from ai_id.

        Returns:
            float: Calculated reward.
        """
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

    def _end_game_reward(self, player_number):
        """Calculates the reward for end-game states.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: End-game reward.
        """
        if self.game.winner is not None:
            is_player = (player_number == 2)  # player_number=2 is black
            winner_is_player = self.game.winner  # True for black, False for white
            if is_player == winner_is_player:
                return (self.end_game_rewards['win_no_timeout']
                        if not getattr(self.game, 'time_up', False)
                        else self.end_game_rewards['win_timeout'])
            return self.end_game_rewards['loss']
        return self.end_game_rewards['draw']

    def _in_game_reward(self, player_number):
        """Calculates the reward for in-game states.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: In-game reward.
        """
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

    def _piece_difference_reward(self, player_number):
        """Calculates reward based on piece count difference.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Piece difference reward.
        """
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

    def _king_bonus_reward(self, player_number):
        """Calculates reward based on king count.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: King bonus reward.
        """
        king_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and abs(piece) == 2:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    king_bonus += 1 if is_player else -1
        return king_bonus

    def _position_bonus_reward(self, player_number):
        """Calculates reward based on piece positions (closer to king row).

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Position bonus reward.
        """
        position_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    distance_to_king = row if player_number == 2 else self.board_size - 1 - row
                    position_bonus += (self.board_size - distance_to_king) if is_player else -(
                        self.board_size - distance_to_king)
        return position_bonus

    def _capture_bonus_reward(self, player_number):
        """Calculates reward for possible captures.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Capture bonus reward.
        """
        capture_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0:
                    moves = get_piece_moves(self.game.board, row, col)
                    if any(len(skipped) > 0 for _, skipped in moves.items()):  # Check for jumps
                        capture_bonus += 1
        return capture_bonus

    def _multi_jump_bonus_reward(self, player_number):
        """Calculates reward for possible multi-jumps.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Multi-jump bonus reward.
        """
        multi_jump_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0:
                    multi_jump_count = self._count_multi_jumps(row, col, set(), player)
                    multi_jump_bonus += multi_jump_count
        return multi_jump_bonus

    def _count_multi_jumps(self, row, col, visited, player):
        """Counts possible multi-jumps for a piece using a temporary board state.

        Args:
            row (int): Current piece row.
            col (int): Current piece column.
            visited (set): Set of visited positions.
            player (int): 1 for white, -1 for black.

        Returns:
            int: Number of multi-jumps.
        """
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        temp_board = self.game.board.copy()
        moves = get_piece_moves(temp_board, row, col)
        multi_jump_count = 0

        for (to_row, to_col), skipped in moves.items():
            if skipped:  # If move is a jump
                move = (row, col, to_row, to_col)
                try:
                    new_board = make_move(temp_board, move)
                    if new_board:
                        multi_jump_count += 1
                        multi_jump_count += self._count_multi_jumps(to_row, to_col, visited.copy(), player)
                    temp_board = self.game.board.copy()  # Reset board for next move
                except CheckersError:
                    continue

        return multi_jump_count

    def _king_capture_bonus_reward(self, player_number):
        """Calculates reward for captures leading to king promotion.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: King capture bonus reward.
        """
        king_capture_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and piece * player > 0 and abs(piece) != 2:
                    moves = get_piece_moves(self.game.board, row, col)
                    for (to_row, to_col), skipped in moves.items():
                        if skipped:  # If move is a jump
                            end_row = to_row
                            if (player_number == 1 and end_row == self.board_size - 1) or (
                                    player_number == 2 and end_row == 0):
                                king_capture_bonus += 1
        return king_capture_bonus

    def _mobility_bonus_reward(self, player_number):
        """Calculates reward for piece mobility.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Mobility bonus reward.
        """
        mobility_bonus = 0
        player = 1 if player_number == 1 else -1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = piece * player > 0
                    moves = get_piece_moves(self.game.board, row, col)
                    mobility_bonus += len(moves) if is_player else -len(moves)
        return mobility_bonus

    def _safety_penalty_reward(self, player_number):
        """Calculates penalty for pieces at risk of being captured.

        Args:
            player_number (int): 1 for white, 2 for black.

        Returns:
            float: Safety penalty reward.
        """
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
                                moves = get_piece_moves(self.game.board, r, c)
                                for (to_row, to_col), skipped in moves.items():
                                    if skipped:  # If move is a jump
                                        mid_row, mid_col = (r + to_row) // 2, (c + to_col) // 2
                                        if (mid_row, mid_col) == (row, col):
                                            safety_penalty -= 1
        return safety_penalty