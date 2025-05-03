# rewards.py
from .constants import BOARD_SIZE
from .config import load_config


class RewardCalculator:
    def __init__(self, game, al_id=None):
        self.game = game
        self.al_id = al_id
        config = load_config()

        # مقادیر پیش‌فرض برای reward_weights
        default_weights = {
            "piece_difference": 1.0,
            "king_bonus": 2.0,
            "position_bonus": 0.1,
            "capture_bonus": 1.0,
            "multi_jump_bonus": 2.0,
            "king_capture_bonus": 3.0,
            "mobility_bonus": 0.1,
            "safety_penalty": -0.5
        }

        # بارگذاری وزن‌های پاداش
        try:
            default_reward_weights = config.get("default_ai_params", {}).get("reward_weights", default_weights)
            self.weights = config.get(f"{al_id}_reward_weights",
                                      default_reward_weights) if al_id else default_reward_weights
        except Exception as e:
            print(f"Error loading reward weights: {e}, using default weights")
            self.weights = default_weights

        # بارگذاری پاداش‌های پایان بازی
        try:
            self.end_game_rewards = config.get("end_game_rewards", {
                "win_no_timeout": 100,
                "win_timeout": 0,
                "draw": -50,
                "loss": -100
            })
        except Exception as e:
            print(f"Error loading end game rewards: {e}, using default end game rewards")
            self.end_game_rewards = {
                "win_no_timeout": 100,
                "win_timeout": 0,
                "draw": -50,
                "loss": -100
            }

    def get_reward(self):
        """محاسبه پاداش کل برای وضعیت فعلی بازی"""
        if self.game.game_over:
            return self._end_game_reward()
        return self._in_game_reward()

    def _end_game_reward(self):
        """پاداش برای حالت پایان بازی"""
        if self.game.winner and not hasattr(self.game, 'time_up'):
            return self.end_game_rewards['win_no_timeout']  # برد بدون اتمام زمان
        elif self.game.winner and hasattr(self.game, 'time_up') and self.game.time_up:
            return self.end_game_rewards['win_timeout']  # برد با اتمام زمان
        elif not self.game.winner:
            return self.end_game_rewards['draw']  # مساوی
        return self.end_game_rewards['loss']  # باخت

    def _in_game_reward(self):
        """پاداش برای حالت‌های میانی بازی"""
        reward = 0
        reward += self._piece_difference_reward() * self.weights['piece_difference']
        reward += self._king_bonus_reward() * self.weights['king_bonus']
        reward += self._position_bonus_reward() * self.weights['position_bonus']
        reward += self._capture_bonus_reward() * self.weights['capture_bonus']
        reward += self._multi_jump_bonus_reward() * self.weights['multi_jump_bonus']
        reward += self._king_capture_bonus_reward() * self.weights['king_capture_bonus']
        reward += self._mobility_bonus_reward() * self.weights['mobility_bonus']
        reward += self._safety_penalty_reward() * self.weights['safety_penalty']
        return reward

    def _piece_difference_reward(self):
        """پاداش برای تفاوت تعداد مهره‌ها"""
        if self.game.turn:
            return self.game.board.player_2_left - self.game.board.player_1_left
        return self.game.board.player_1_left - self.game.board.player_2_left

    def _king_bonus_reward(self):
        """پاداش برای شاه‌ها"""
        king_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.king:
                    if piece.is_player_2 == self.game.turn:
                        king_bonus += 1
                    else:
                        king_bonus -= 1
        return king_bonus

    def _position_bonus_reward(self):
        """پاداش برای موقعیت مهره‌ها (نزدیکی به ردیف شاه شدن)"""
        position_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn:
                    if piece.is_player_2:
                        distance_to_king = row  # برای بازیکن 2، ردیف 7
                    else:
                        distance_to_king = BOARD_SIZE - 1 - row  # برای بازیکن 1، ردیف 0
                    position_bonus += (BOARD_SIZE - distance_to_king)
                elif piece:
                    if piece.is_player_2:
                        distance_to_king = row
                    else:
                        distance_to_king = BOARD_SIZE - 1 - row
                    position_bonus -= (BOARD_SIZE - distance_to_king)
        return position_bonus

    def _capture_bonus_reward(self):
        """پاداش برای پرش‌های ممکن"""
        capture_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn:
                    moves = self.game.get_valid_moves(piece)
                    if any(skipped for skipped in moves.values()):
                        capture_bonus += 1
        return capture_bonus

    def _multi_jump_bonus_reward(self):
        """پاداش برای پرش‌های چندگانه ممکن"""
        multi_jump_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn:
                    multi_jump_count = self._count_multi_jumps(piece, row, col, set())
                    multi_jump_bonus += multi_jump_count
        return multi_jump_bonus

    def _count_multi_jumps(self, piece, row, col, visited):
        """محاسبه تعداد پرش‌های چندگانه ممکن برای یه مهره"""
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        moves = self.game.get_valid_moves(piece)
        multi_jump_count = 0
        for (next_row, next_col), skipped in moves.items():
            if skipped:  # اگه حرکت پرش باشه
                # شبیه‌سازی پرش
                original_position = (piece.row, piece.col)
                piece.move(next_row, next_col)
                multi_jump_count += 1
                # بررسی پرش‌های بعدی
                multi_jump_count += self._count_multi_jumps(piece, next_row, next_col, visited.copy())
                # بازگرداندن مهره به موقعیت اصلی
                piece.move(original_position[0], original_position[1])
        return multi_jump_count

    def _king_capture_bonus_reward(self):
        """پاداش برای پرش‌هایی که منجر به شاه شدن می‌شن"""
        king_capture_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn and not piece.king:
                    moves = self.game.get_valid_moves(piece)
                    for (next_row, next_col), skipped in moves.items():
                        if skipped:
                            if (piece.is_player_2 and next_row == BOARD_SIZE - 1) or \
                                    (not piece.is_player_2 and next_row == 0):
                                king_capture_bonus += 1
        return king_capture_bonus

    def _mobility_bonus_reward(self):
        """پاداش برای تحرک‌پذیری مهره‌ها"""
        mobility_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn:
                    moves = self.game.get_valid_moves(piece)
                    mobility_bonus += len(moves)
                elif piece:
                    moves = self.game.get_valid_moves(piece)
                    mobility_bonus -= len(moves)
        return mobility_bonus

    def _safety_penalty_reward(self):
        """جریمه برای مهره‌های در خطر پرش"""
        safety_penalty = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row][col]
                if piece and piece.is_player_2 == self.game.turn:
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            opponent = self.game.board.board[r][c]
                            if opponent and opponent.is_player_2 != self.game.turn:
                                opponent_moves = self.game.get_valid_moves(opponent)
                                if any((row, col) in move for move, skipped in opponent_moves.items() if skipped):
                                    safety_penalty -= 1
        return safety_penalty