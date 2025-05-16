import json
from pathlib import Path


class RewardCalculator:
    def __init__(self, game, ai_id=None):
        """
        مقداردهی اولیه محاسبه‌گر پاداش.

        Args:
            game: شیء بازی (Game)
            ai_id: شناسه AI (مثل 'ai_1' یا 'ai_2')، اگر None باشد از تنظیمات پیش‌فرض استفاده می‌شود
        """
        self.game = game
        self.ai_id = ai_id
        self.board_size = self.game.board_size  # استفاده از board_size از شیء game

        # مقادیر پیش‌فرض برای وزن‌های پاداش و پاداش‌های پایان بازی
        self.default_weights = {
            "piece_difference": 1.0,
            "king_bonus": 2.0,
            "position_bonus": 0.1,
            "capture_bonus": 1.0,
            "multi_jump_bonus": 2.0,
            "king_capture_bonus": 3.0,
            "mobility_bonus": 0.1,
            "safety_penalty": -0.5
        }
        self.default_end_game_rewards = {
            "win_no_timeout": 100,
            "win_timeout": 0,
            "draw": -50,
            "loss": -100
        }

        # بارگذاری تنظیمات
        self.weights, self.end_game_rewards = self._load_config()

    def _load_config(self):
        """
        بارگذاری وزن‌های پاداش و پاداش‌های پایان بازی از فایل کانفیگ خاص AI.

        Returns:
            tuple: (وزن‌های پاداش, پاداش‌های پایان بازی)
        """
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        config_path = Path(__file__).parent.parent / "configs" / "ai_config.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                ai_config = json.load(f)

            # پیدا کردن ai_code برای این ai_id
            ai_code = ai_config["ai_configs"].get(player_key, {}).get("ai_code")
            if not ai_code:
                print(f"Warning: No ai_code found for {player_key}, using default weights")
                return self.default_weights, self.default_end_game_rewards

            # بارگذاری کانفیگ خاص AI
            specific_config_path = Path(__file__).parent.parent / "configs" / "ai" / f"{ai_code}_config.json"
            if specific_config_path.exists():
                with open(specific_config_path, "r", encoding="utf-8") as f:
                    specific_config = json.load(f)
                params = specific_config.get(player_key, {})
                weights = params.get("reward_weights", self.default_weights)
                end_game_rewards = params.get("end_game_rewards", self.default_end_game_rewards)
                print(f"Loaded config for {player_key} from {specific_config_path}")
                return weights, end_game_rewards
            else:
                print(f"Warning: Config file {specific_config_path} not found, using default weights")
                return self.default_weights, self.default_end_game_rewards
        except Exception as e:
            print(f"Error loading config: {e}, using default weights")
            return self.default_weights, self.default_end_game_rewards

    def get_reward(self, player_number=None):
        """
        محاسبه پاداش کل برای وضعیت فعلی بازی.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه. اگر None باشد، از ai_id استفاده می‌شود.

        Returns:
            float: مقدار پاداش محاسبه‌شده
        """
        if player_number is None and self.ai_id:
            player_number = 1 if self.ai_id == "ai_1" else 2
        if player_number not in [1, 2]:
            print(f"Warning: Invalid player_number {player_number}, returning 0 reward")
            return 0.0

        if self.game.game_over:
            return self._end_game_reward(player_number)
        return self._in_game_reward(player_number)

    def _end_game_reward(self, player_number):
        """
        محاسبه پاداش برای حالت پایان بازی.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش پایان بازی
        """
        if self.game.winner is not None:
            is_player = (player_number == 2)  # player_number=2 سیاه
            winner_is_player = self.game.winner  # True برای سیاه، False برای سفید
            if is_player == winner_is_player:
                return (self.end_game_rewards['win_no_timeout']
                        if not getattr(self.game, 'time_up', False)
                        else self.end_game_rewards['win_timeout'])
            return self.end_game_rewards['loss']
        return self.end_game_rewards['draw']

    def _in_game_reward(self, player_number):
        """
        محاسبه پاداش برای حالت‌های میانی بازی.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش میانی
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
        """
        محاسبه پاداش برای تفاوت تعداد مهره‌ها.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش تفاوت مهره‌ها
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
        """
        محاسبه پاداش برای تعداد شاه‌ها.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش شاه‌ها
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
        """
        محاسبه پاداش برای موقعیت مهره‌ها (نزدیکی به ردیف شاه شدن).

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش موقعیت
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
        """
        محاسبه پاداش برای پرش‌های ممکن.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش پرش‌های ممکن
        """
        capture_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    moves = self.game.get_valid_moves(row, col)
                    if any(skipped for skipped in moves.values()):
                        capture_bonus += 1
        return capture_bonus

    def _multi_jump_bonus_reward(self, player_number):
        """
        محاسبه پاداش برای پرش‌های چندگانه ممکن.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش پرش‌های چندگانه
        """
        multi_jump_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    multi_jump_count = self._count_multi_jumps(row, col, set(), player_number)
                    multi_jump_bonus += multi_jump_count
        return multi_jump_bonus

    def _count_multi_jumps(self, row, col, visited, player_number):
        """
        محاسبه تعداد پرش‌های چندگانه ممکن برای یک مهره.

        Args:
            row: ردیف فعلی مهره
            col: ستون فعلی مهره
            visited: مجموعه موقعیت‌های بازدیدشده
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            int: تعداد پرش‌های چندگانه
        """
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        moves = self.game.get_valid_moves(row, col)
        multi_jump_count = 0

        for move, skipped in moves.items():
            if skipped:  # اگر حرکت پرش باشد
                start_row, start_col, end_row, end_col = move  # فرمت: (start_row, start_col, end_row, end_col)
                # شبیه‌سازی پرش در تخته
                original_piece = self.game.board.board[row, col]
                original_target = self.game.board.board[end_row, end_col]
                original_skipped = [(r, c) for r, c in skipped] if skipped else []

                # اعمال پرش
                self.game.board.board[row, col] = 0
                self.game.board.board[end_row, end_col] = original_piece
                for r, c in original_skipped:
                    self.game.board.board[r, c] = 0

                multi_jump_count += 1
                # بررسی پرش‌های بعدی
                multi_jump_count += self._count_multi_jumps(end_row, end_col, visited.copy(), player_number)

                # بازگرداندن تخته به حالت اولیه
                self.game.board.board[row, col] = original_piece
                self.game.board.board[end_row, end_col] = original_target
                for r, c in original_skipped:
                    self.game.board.board[r, c] = (
                        -original_piece if (original_piece > 0 and player_number == 2) or
                                           (original_piece < 0 and player_number == 1) else original_piece
                    )

        return multi_jump_count

    def _king_capture_bonus_reward(self, player_number):
        """
        محاسبه پاداش برای پرش‌هایی که منجر به شاه شدن می‌شوند.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش پرش‌های منجر به شاه شدن
        """
        king_capture_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)) and abs(
                        piece) != 2:
                    moves = self.game.get_valid_moves(row, col)
                    for move, skipped in moves.items():
                        if skipped:
                            _, _, end_row, _ = move  # فقط end_row نیاز است
                            if (player_number == 1 and end_row == 0) or (
                                    player_number == 2 and end_row == self.board_size - 1):
                                king_capture_bonus += 1
        return king_capture_bonus

    def _mobility_bonus_reward(self, player_number):
        """
        محاسبه پاداش برای تحرک‌پذیری مهره‌ها.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: پاداش تحرک‌پذیری
        """
        mobility_bonus = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0:
                    is_player = (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                    moves = self.game.get_valid_moves(row, col)
                    mobility_bonus += len(moves) if is_player else -len(moves)
        return mobility_bonus

    def _safety_penalty_reward(self, player_number):
        """
        محاسبه جریمه برای مهره‌های در خطر پرش.

        Args:
            player_number: 1 برای سفید، 2 برای سیاه

        Returns:
            float: جریمه ایمنی
        """
        safety_penalty = 0
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    for r in range(self.board_size):
                        for c in range(self.board_size):
                            opponent = self.game.board.board[r, c]
                            if opponent != 0 and (
                                    (opponent > 0 and player_number == 2) or (opponent < 0 and player_number == 1)):
                                opponent_moves = self.game.get_valid_moves(r, c)
                                if any((row, col) in skipped for move, skipped in opponent_moves.items() if skipped):
                                    safety_penalty -= 1
        return safety_penalty