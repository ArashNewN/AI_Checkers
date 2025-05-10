from .constants import BOARD_SIZE
from .config import load_config

class RewardCalculator:
    def __init__(self, game, ai_id=None):
        self.game = game
        self.ai_id = ai_id
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
            self.weights = config.get(f"{ai_id}_reward_weights",
                                     default_reward_weights) if ai_id else default_reward_weights
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

    def get_reward(self, player_number=None):
        """
        محاسبه پاداش کل برای وضعیت فعلی بازی.
        player_number: 1 برای سفید، 2 برای سیاه. اگه None باشه، از ai_id استفاده می‌شه.
        """
        if player_number is None and self.ai_id:
            player_number = 1 if self.ai_id == "ai_1" else 2

        if self.game.game_over:
            return self._end_game_reward(player_number)
        return self._in_game_reward(player_number)

    def _end_game_reward(self, player_number):
        """پاداش برای حالت پایان بازی"""
        if self.game.winner:
            is_player = player_number == 2  # فرض: player_number=2 سیاهه
            winner_is_player = self.game.winner == "black"
            if is_player == winner_is_player:
                return self.end_game_rewards['win_no_timeout'] if not getattr(self.game, 'time_up', False) else \
                       self.end_game_rewards['win_timeout']
            return self.end_game_rewards['loss']
        return self.end_game_rewards['draw']

    def _in_game_reward(self, player_number):
        """پاداش برای حالت‌های میانی بازی"""
        reward = 0
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
        """پاداش برای تفاوت تعداد مهره‌ها"""
        player_pieces = sum(
            1 for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
            if (self.game.board.board[row, col] > 0 and player_number == 1) or
               (self.game.board.board[row, col] < 0 and player_number == 2))
        opponent_pieces = sum(
            1 for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)
            if (self.game.board.board[row, col] > 0 and player_number == 2) or
               (self.game.board.board[row, col] < 0 and player_number == 1))
        return player_pieces - opponent_pieces

    def _king_bonus_reward(self, player_number):
        """پاداش برای شاه‌ها"""
        king_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0 and abs(piece) == 2:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    king_bonus += 1 if is_player else -1
        return king_bonus

    def _position_bonus_reward(self, player_number):
        """پاداش برای موقعیت مهره‌ها (نزدیکی به ردیف شاه شدن)"""
        position_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    distance_to_king = row if player_number == 2 else BOARD_SIZE - 1 - row
                    position_bonus += (BOARD_SIZE - distance_to_king) if is_player else -(BOARD_SIZE - distance_to_king)
        return position_bonus

    def _capture_bonus_reward(self, player_number):
        """پاداش برای پرش‌های ممکن"""
        capture_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    moves = self.game.get_valid_moves(row, col)
                    if any(skipped for skipped in moves.values()):
                        capture_bonus += 1
        return capture_bonus

    def _multi_jump_bonus_reward(self, player_number):
        """پاداش برای پرش‌های چندگانه ممکن"""
        multi_jump_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    multi_jump_count = self._count_multi_jumps(row, col, set(), player_number)
                    multi_jump_bonus += multi_jump_count
        return multi_jump_bonus

    def _count_multi_jumps(self, row, col, visited, player_number):
        """محاسبه تعداد پرش‌های چندگانه ممکن برای یک مهره"""
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        moves = self.game.get_valid_moves(row, col)
        multi_jump_count = 0
        for (next_row, next_col), skipped in moves.items():
            if skipped:  # اگر حرکت پرش باشد
                # شبیه‌سازی پرش در تخته
                original_piece = self.game.board.board[row, col]
                original_target = self.game.board.board[next_row, next_col]
                original_skipped = [(r, c) for r, c in skipped] if skipped else []

                # اعمال پرش
                self.game.board.board[row, col] = 0
                self.game.board.board[next_row, next_col] = original_piece
                for r, c in original_skipped:
                    self.game.board.board[r, c] = 0

                multi_jump_count += 1
                # بررسی پرش‌های بعدی
                multi_jump_count += self._count_multi_jumps(next_row, next_col, visited.copy(), player_number)

                # بازگرداندن تخته به حالت اولیه
                self.game.board.board[row, col] = original_piece
                self.game.board.board[next_row, next_col] = original_target
                for r, c in original_skipped:
                    self.game.board.board[r, c] = self.game.board.board[r, c] or -original_piece

        return multi_jump_count

    def _king_capture_bonus_reward(self, player_number):
        """پاداش برای پرش‌هایی که منجر به شاه شدن می‌شوند"""
        king_capture_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)) and abs(piece) != 2:
                    moves = self.game.get_valid_moves(row, col)
                    for (next_row, next_col), skipped in moves.items():
                        if skipped:
                            if (player_number == 2 and next_row == BOARD_SIZE - 1) or (player_number == 1 and next_row == 0):
                                king_capture_bonus += 1
        return king_capture_bonus

    def _mobility_bonus_reward(self, player_number):
        """پاداش برای تحرک‌پذیری مهره‌ها"""
        mobility_bonus = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    moves = self.game.get_valid_moves(row, col)
                    mobility_bonus += len(moves) if is_player else -len(moves)
        return mobility_bonus

    def _safety_penalty_reward(self, player_number):
        """جریمه برای مهره‌های در خطر پرش"""
        safety_penalty = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            opponent = self.game.board.board[r, c]
                            if opponent != 0 and ((opponent > 0 and player_number == 2) or (opponent < 0 and player_number == 1)):
                                opponent_moves = self.game.get_valid_moves(r, c)
                                if any((row, col) in skipped for move, skipped in opponent_moves.items() if skipped):
                                    safety_penalty -= 1
        return safety_penalty