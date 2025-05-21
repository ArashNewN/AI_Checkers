import pygame
import os
import random
from .board import Board
from .timer import TimerManager
from .ai import AdvancedAI
from .alphazero_ai import AlphaZeroAI
from .config import load_stats, save_stats
from .constants import WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE, BOARD_WIDTH, MENU_HEIGHT, BORDER_THICKNESS, SQUARE_SIZE, RED

class Game:
    def __init__(self, settings, interface=None):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Checkers: Player 1 vs Player 2')
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
        self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)
        self.settings = settings
        self.interface = interface
        stats = load_stats()
        self.player_1_wins = stats["player_1_wins"]
        self.player_2_wins = stats["player_2_wins"]
        self.player_1_ai = None
        self.player_2_ai = None
        self.update_ai_players()
        self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()
        self.init_game()

    def update_ai_players(self):
        # هماهنگی game_mode با نوع AI
        # بررسی حالت بازی برای تنظیم نوع AI برای هر بازیکن
        if self.settings.game_mode == "human_vs_human":
            # اگه حالت بازی انسانی در برابر انسانی باشه
            self.settings.player_1_ai_type = "none"  # بازیکن 1 هوش مصنوعی نداره
            self.settings.player_2_ai_type = "none"  # بازیکن 2 هوش مصنوعی نداره
        elif self.settings.game_mode == "human_vs_ai":
            # اگه حالت بازی انسانی در برابر هوش مصنوعی باشه
            self.settings.player_1_ai_type = "none"  # بازیکن 1 هوش مصنوعی نداره
            # player_2_ai_type می‌تونه advanced یا alphazero باشه
            # بررسی نوع هوش مصنوعی بازیکن 2
            if self.settings.player_2_ai_type == "none":
                # اگه نوع هوش مصنوعی بازیکن 2 مشخص نشده باشه
                self.settings.player_2_ai_type = "advanced"  # پیش‌فرض: هوش مصنوعی پیشرفته
        elif self.settings.game_mode == "ai_vs_ai":
            # اگه حالت بازی هوش مصنوعی در برابر هوش مصنوعی باشه
            # هر دو باید AI باشن
            # بررسی نوع هوش مصنوعی بازیکن 1
            if self.settings.player_1_ai_type == "none":
                # اگه نوع هوش مصنوعی بازیکن 1 مشخص نشده باشه
                self.settings.player_1_ai_type = "advanced"  # پیش‌فرض: هوش مصنوعی پیشرفته
            # بررسی نوع هوش مصنوعی بازیکن 2
            if self.settings.player_2_ai_type == "none":
                # اگه نوع هوش مصنوعی بازیکن 2 مشخص نشده باشه
                self.settings.player_2_ai_type = "advanced"  # پیش‌فرض: هوش مصنوعی پیشرفته

        # تخصیص AI بر اساس تنظیمات player_1_ai_type و player_2_ai_type
        # تنظیم هوش مصنوعی برای بازیکن 1
        if self.settings.player_1_ai_type == "advanced":
            # اگه نوع هوش مصنوعی بازیکن 1 پیشرفته باشه
            self.player_1_ai = AdvancedAI(self.settings.player_1_color, self.settings.ai_ability,
                                          "ai_model.pth")  # ایجاد شیء هوش مصنوعی پیشرفته
        elif self.settings.player_1_ai_type == "alphazero":
            # اگه نوع هوش مصنوعی بازیکن 1 آلفازیرو باشه
            self.player_1_ai = AlphaZeroAI(self.settings.player_1_color,
                                           model_path="alphazero_model.pth")  # ایجاد شیء هوش مصنوعی آلفازیرو
        else:
            # اگه هوش مصنوعی برای بازیکن 1 غیرفعال باشه
            self.player_1_ai = None  # هیچ هوش مصنوعی برای بازیکن 1 تنظیم نمی‌شه

        # تنظیم هوش مصنوعی برای بازیکن 2
        if self.settings.player_2_ai_type == "advanced":
            # اگه نوع هوش مصنوعی بازیکن 2 پیشرفته باشه
            self.player_2_ai = AdvancedAI(self.settings.player_2_color, self.settings.ai_ability,
                                          "ai_model1.pth")  # ایجاد شیء هوش مصنوعی پیشرفته
        elif self.settings.player_2_ai_type == "alphazero":
            # اگه نوع هوش مصنوعی بازیکن 2 آلفازیرو باشه
            self.player_2_ai = AlphaZeroAI(self.settings.player_2_color,
                                           model_path="alphazero_model.pth")  # ایجاد شیء هوش مصنوعی آلفازیرو
        else:
            # اگه هوش مصنوعی برای بازیکن 2 غیرفعال باشه
            self.player_2_ai = None  # هیچ هوش مصنوعی برای بازیکن 2 تنظیم نمی‌شه

    def init_game(self):
        # مقداردهی اولیه بازی
        self.board = Board(self.settings)  # ایجاد یه صفحه بازی جدید با تنظیمات مشخص‌شده
        self.selected = None  # هیچ مهره‌ای انتخاب نشده
        self.turn = False if self.settings.player_starts else True  # تنظیم نوبت اولیه (False برای بازیکن 1، True برای بازیکن 2)
        self.valid_moves = {}  # دیکشنری حرکت‌های معتبر خالیه
        self.game_over = False  # بازی هنوز تموم نشده
        self.winner = None  # هنوز برنده‌ای وجود نداره
        self.last_state = None  # آخرین حالت بازی (برای AI) خالیه
        self.last_action = None  # آخرین حرکت (برای AI) خالیه
        self.move_log = []  # لیست لاگ حرکت‌ها خالیه
        self.click_log = []  # لیست لاگ کلیک‌ها خالیه
        self.game_started = False  # بازی هنوز شروع نشده
        self.score_updated = False  # امتیازها هنوز آپدیت نشدن
        self.no_capture_or_king_moves = 0  # تعداد حرکت‌های بدون پرش یا شاه شدن صفره
        self.timer.start_game()  # شروع تایمر بازی
        self.last_update_time = pygame.time.get_ticks()  # ثبت زمان فعلی برای آپدیت‌ها
        self.repeat_count = 0  # تعداد تکرار بازی (برای حالت AI) صفره

    def start_game(self):
        # شروع رسمی بازی
        self.game_started = True  # بازی شروع شده
        self.game_over = False  # بازی تموم نشده
        self.winner = None  # هنوز برنده‌ای وجود نداره
        self.selected = None  # ریست انتخاب مهره
        self.valid_moves = {}  # ریست حرکت‌های معتبر
        self.multi_jump_active = False  # ریست پرچم پرش چندگانه
        self.last_update_time = pygame.time.get_ticks()  # ثبت زمان فعلی برای آپدیت‌ها
        self.timer.start_game()  # شروع تایمر بازی
        self.update_ai_players()  # آپدیت تنظیمات هوش مصنوعی
        # بررسی حالت بازی برای تنظیم تعداد تکرار
        if self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game":
            # اگه حالت بازی AI در برابر AI باشه و حالت تکرار فعال باشه
            self.repeat_count = self.settings.repeat_hands  # تعداد تکرار بازی رو از تنظیمات بگیر
        else:
            # در غیر این صورت
            self.repeat_count = 1  # بازی فقط یه بار اجرا می‌شه

    def reset_board(self):
        # ریست کردن صفحه بازی
        self.board = Board(self.settings)  # ایجاد یه صفحه بازی جدید
        self.selected = None  # هیچ مهره‌ای انتخاب نشده
        self.turn = False if self.settings.player_starts else True  # تنظیم نوبت اولیه
        self.valid_moves = {}  # دیکشنری حرکت‌های معتبر خالیه
        self.game_over = False  # بازی تموم نشده
        self.winner = None  # هنوز برنده‌ای وجود نداره
        self.move_log = []  # لاگ حرکت‌ها خالیه
        self.click_log = []  # لاگ کلیک‌ها خالیه
        self.no_capture_or_king_moves = 0  # تعداد حرکت‌های بدون پرش یا شاه شدن صفره
        self.score_updated = False  # امتیازها هنوز آپدیت نشدن
        self.multi_jump_active = False  # ریست پرچم پرش چندگانه
        self.timer.start_game()  # شروع دوباره تایمر
        self.last_update_time = pygame.time.get_ticks()  # ثبت زمان فعلی
        self.update_ai_players()  # آپدیت تنظیمات هوش مصنوعی

    def _update_game(self):
        # آپدیت وضعیت بازی
        if self.game_started and not self.game_over:
            # اگه بازی شروع شده باشه و تموم نشده باشه
            self.timer.update(self.turn, self.game_started, self.game_over)  # آپدیت تایمر با توجه به نوبت و وضعیت بازی
            self.check_winner()  # بررسی اینکه آیا برنده‌ای وجود داره

    def check_winner(self):
        # بررسی وضعیت برنده بازی
        if not self.game_started or self.game_over:
            # اگه بازی شروع نشده باشه یا تموم شده باشه، چیزی برنگردون
            return
        player_1_moves = False  # پرچم برای وجود حرکت برای بازیکن 1
        player_2_moves = False  # پرچم برای وجود حرکت برای بازیکن 2
        # حلقه برای بررسی همه خانه‌های صفحه
        for row in range(BOARD_SIZE):
            # برای هر ردیف
            for col in range(BOARD_SIZE):
                # برای هر ستون
                piece = self.board.board[row][col]  # گرفتن مهره تو موقعیت فعلی
                if piece:
                    # اگه مهره‌ای وجود داشته باشه
                    moves = self.get_valid_moves(piece)  # گرفتن حرکت‌های معتبر برای این مهره
                    if moves:
                        # اگه حرکت‌های معتبری وجود داشته باشه
                        if not piece.is_player_2:
                            # اگه مهره متعلق به بازیکن 1 باشه
                            player_1_moves = True  # بازیکن 1 حرکت داره
                        else:
                            # اگه مهره متعلق به بازیکن 2 باشه
                            player_2_moves = True  # بازیکن 2 حرکت داره
        if not player_1_moves:
            # اگه بازیکن 1 هیچ حرکتی نداشته باشه
            self.winner = True  # بازیکن 2 برنده‌ست
            self.game_over = True  # بازی تموم شده
        elif not player_2_moves:
            # اگه بازیکن 2 هیچ حرکتی نداشته باشه
            self.winner = False  # بازیکن 1 برنده‌ست
            self.game_over = True  # بازی تموم شده
        elif self.no_capture_or_king_moves >= 40:
            # اگه تعداد حرکت‌های بدون پرش یا شاه شدن به 40 برسه
            self.winner = None  # بازی مساویه
            self.game_over = True  # بازی تموم شده
        if self.settings.use_timer:
            # اگه تایمر فعال باشه
            player_1_time = self.timer.get_current_time(False)  # گرفتن زمان باقی‌مونده بازیکن 1
            player_2_time = self.timer.get_current_time(True)  # گرفتن زمان باقی‌مونده بازیکن 2
            if player_1_time <= 0:
                # اگه زمان بازیکن 1 تموم شده باشه
                self.winner = True  # بازیکن 2 برنده‌ست
                self.game_over = True  # بازی تموم شده
                self.time_up = True  # پرچم اتمام زمان فعال می‌شه
            elif player_2_time <= 0:
                # اگه زمان بازیکن 2 تموم شده باشه
                self.winner = False  # بازیکن 1 برنده‌ست
                self.game_over = True  # بازی تموم شده
                self.time_up = True  # پرچم اتمام زمان فعال می‌شه
        if self.game_over and not self.score_updated:
            # اگه بازی تموم شده باشه و امتیازها آپدیت نشده باشن
            self.score_updated = True  # پرچم آپدیت امتیاز فعال می‌شه
            if self.winner is not None:
                # اگه برنده‌ای وجود داشته باشه
                if not self.winner:
                    # اگه برنده بازیکن 1 باشه
                    self.player_1_wins += 1  # افزایش امتیاز بازیکن 1
                else:
                    # اگه برنده بازیکن 2 باشه
                    self.player_2_wins += 1  # افزایش امتیاز بازیکن 2
            save_stats({"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins})  # ذخیره آمار بازی

    def make_ai_move(self, ai, current_state=None):
        # اجرای حرکت توسط هوش مصنوعی
        if not ai:
            # اگه هوش مصنوعی وجود نداشته باشه، چیزی برنگردون
            return
        valid_pieces = []  # لیست مهره‌هایی که حرکت معتبر دارن
        # حلقه برای بررسی همه خانه‌های صفحه
        for row in range(BOARD_SIZE):
            # برای هر ردیف
            for col in range(BOARD_SIZE):
                # برای هر ستون
                piece = self.board.board[row][col]  # گرفتن مهره تو موقعیت فعلی
                if piece and piece.is_player_2 == self.turn:
                    # اگه مهره متعلق به بازیکن فعلی باشه
                    moves = self.get_valid_moves(piece)  # گرفتن حرکت‌های معتبر برای این مهره
                    if moves:
                        # اگه حرکت‌های معتبری وجود داشته باشه
                        valid_pieces.append((piece, moves))  # اضافه کردن مهره و حرکت‌هاش به لیست
        if valid_pieces:
            # اگه مهره‌هایی با حرکت معتبر وجود داشته باشن
            piece, moves = random.choice(valid_pieces)  # انتخاب تصادفی یه مهره و حرکت‌هاش
            valid_moves = moves  # ذخیره حرکت‌های معتبر
            move = ai.act(self, valid_moves)  # گرفتن حرکت پیشنهادی از هوش مصنوعی
            if move and piece:
                # اگه حرکت و مهره معتبر باشن
                self.last_state = current_state  # ذخیره حالت فعلی (برای AI)
                self.last_action = move  # ذخیره حرکت انتخاب‌شده
                self.selected = piece  # انتخاب مهره
                self.valid_moves = valid_moves  # تنظیم حرکت‌های معتبر
                if self._move(move[0], move[1]):
                    # اگه حرکت با موفقیت انجام بشه
                    self.move_log.append((piece.row, piece.col, move[0], move[1]))  # اضافه کردن حرکت به لاگ
                    if len(self.move_log) > 30:
                        # اگه لاگ حرکت‌ها بیشتر از 30 تا بشه
                        self.move_log.pop(0)  # حذف قدیمی‌ترین حرکت
                if self.game_over and isinstance(ai, AdvancedAI):
                    # اگه بازی تموم شده باشه و هوش مصنوعی از نوع پیشرفته باشه
                    ai.update_target_network()  # آپدیت شبکه هدف هوش مصنوعی
                    ai.save_model()  # ذخیره مدل هوش مصنوعی
                if self.settings.sound_enabled and os.path.exists('move.wav') and os.path.getsize('move.wav') > 0:
                    # اگه صدا فعال باشه و فایل صوتی موجود باشه
                    try:
                        pygame.mixer.Sound('move.wav').play()  # پخش صدای حرکت
                    except pygame.error as e:
                        # اگه خطایی تو پخش صدا پیش بیاد
                        print(f"Error playing move.wav: {e}")  # چاپ خطا

    def get_reward(self):
        # محاسبه پاداش برای هوش مصنوعی
        if self.game_over:
            # اگه بازی تموم شده باشه
            if self.winner and not hasattr(self, 'time_up'):
                # اگه برنده وجود داشته باشه و اتمام زمان نباشه
                return 100  # پاداش مثبت برای برد
            elif self.winner and hasattr(self, 'time_up') and self.time_up:
                # اگه برنده وجود داشته باشه و به‌خاطر اتمام زمان باشه
                return 0  # بدون پاداش
            else:
                # اگه بازنده باشه
                return -100  # پاداش منفی برای باخت
        # محاسبه تفاوت تعداد مهره‌ها
        piece_difference = self.board.player_2_left - self.board.player_1_left if self.turn else self.board.player_1_left - self.board.player_2_left
        king_bonus = 0  # پاداش برای مهره‌های شاه
        # حلقه برای بررسی همه خانه‌های صفحه
        for row in range(BOARD_SIZE):
            # برای هر ردیف
            for col in range(BOARD_SIZE):
                # برای هر ستون
                piece = self.board.board[row][col]  # گرفتن مهره تو موقعیت فعلی
                if piece and piece.king:
                    # اگه مهره‌ای وجود داشته باشه و شاه باشه
                    if piece.is_player_2 == self.turn:
                        # اگه مهره متعلق به بازیکن فعلی باشه
                        king_bonus += 0.5  # اضافه کردن پاداش برای شاه
                    else:
                        # اگه مهره متعلق به حریف باشه
                        king_bonus -= 0.5  # کم کردن پاداش برای شاه حریف
        return piece_difference + king_bonus  # برگردوندن پاداش نهایی

    def handle_click(self, pos):
        # پردازش کلیک کاربر
        x, y = pos  # گرفتن مختصات کلیک (x, y)
        self.click_log.append((x, y))  # اضافه کردن مختصات کلیک به لاگ
        if len(self.click_log) > 50:
            # اگه تعداد کلیک‌ها بیشتر از 50 بشه
            self.click_log.pop(0)  # حذف قدیمی‌ترین کلیک
        if x < BOARD_WIDTH and self.game_started and not self.game_over:
            # اگه کلیک تو محدوده صفحه بازی باشه و بازی شروع شده و تموم نشده باشه
            if self.settings.game_mode == "human_vs_human" or (
                    self.settings.game_mode == "human_vs_ai" and not self.turn):
                # اگه حالت بازی انسانی باشه یا تو حالت انسانی در برابر AI نوبت بازیکن انسانی باشه
                row = (y - MENU_HEIGHT - BORDER_THICKNESS) // SQUARE_SIZE  # محاسبه ردیف کلیک‌شده
                col = (x - BORDER_THICKNESS) // SQUARE_SIZE  # محاسبه ستون کلیک‌شده
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    # اگه مختصات کلیک خارج از صفحه بازی باشه
                    return False  # برگردوندن False
                if self.selected and (row, col) in self.valid_moves:
                    # اگه یه مهره انتخاب شده باشه و مختصات کلیک تو حرکت‌های معتبر باشه
                    return self._move(row, col)  # اجرای حرکت
                # تلاش برای انتخاب مهره
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    # اگه انتخاب ناموفق باشه و پرش چندگانه فعال باشه
                    self.valid_moves = self.get_valid_moves(self.selected)  # بازگرداندن حرکت‌های معتبر برای مهره فعلی
                return result  # برگردوندن نتیجه انتخاب
        # اگه بازی تموم شده باشه یا شروع نشده باشه، متغیرها رو ریست کن
        if self.game_over:
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
        return False  # اگه شرایط بالا برقرار نباشه، برگردوندن False

    def select(self, row, col):
        # انتخاب یه مهره
        if self.game_over or not self.game_started:
            # اگه بازی تموم شده باشه یا شروع نشده باشه
            return False  # هیچ انتخابی انجام نشه
        if self.selected:
            # اگه قبلاً یه مهره انتخاب شده باشه
            result = self._move(row, col)  # تلاش برای حرکت به مختصات کلیک‌شده
            if not result:
                # اگه حرکت ناموفق باشه
                if not self.multi_jump_active:
                    # اگه پرش چندگانه فعال نباشه
                    self.selected = None  # ریست انتخاب مهره
                    return self.select(row, col)  # تلاش دوباره برای انتخاب
                return False  # تو حالت پرش چندگانه، انتخاب جدید غیرمجازه
        piece = self.board.board[row][col]  # گرفتن مهره تو مختصات کلیک‌شده
        if piece and (piece.is_player_2 == self.turn):
            # اگه مهره‌ای وجود داشته باشه و متعلق به بازیکن فعلی باشه
            if self.multi_jump_active and piece != self.selected:
                # اگه حالت پرش چندگانه فعال باشه و مهره جدید با مهره انتخاب‌شده فرق کنه
                return False  # جلوگیری از انتخاب مهره دیگه
            self.selected = piece  # انتخاب مهره
            self.valid_moves = self.get_valid_moves(piece)  # گرفتن حرکت‌های معتبر برای مهره
            return True  # برگردوندن True برای انتخاب موفق
        return False  # اگه مهره‌ای نباشه یا متعلق به بازیکن فعلی نباشه، برگردوندن False

    def _move(self, row, col):
        # اعمال حرکت یا پرش
        if not self.selected or (row, col) not in self.valid_moves:
            # اگه مهره‌ای انتخاب نشده باشه یا مختصات مقصد تو حرکت‌های معتبر نباشه
            return False  # برگردوندن False
        piece = self.board.board[self.selected.row][self.selected.col]  # گرفتن مهره انتخاب‌شده
        if piece:
            # اگه مهره‌ای وجود داشته باشه
            start_row, start_col = self.selected.row, self.selected.col  # ذخیره موقعیت اولیه مهره
            self.board.board[self.selected.row][self.selected.col] = None  # خالی کردن خانه اولیه
            skipped = self.valid_moves[(row, col)]  # گرفتن لیست مهره‌های حذف‌شده (برای پرش)
            if skipped:
                # اگه پرش انجام شده باشه
                self.board.remove(skipped)  # حذف مهره‌های حریف
                self.no_capture_or_king_moves = 0  # ریست شمارش حرکت‌های بدون پرش
            else:
                # اگه حرکت ساده باشه
                self.no_capture_or_king_moves += 1  # افزایش شمارش حرکت‌های بدون پرش
            piece.move(row, col)  # جابه‌جایی مهره به موقعیت جدید
            self.board.board[row][col] = piece  # قرار دادن مهره تو خانه جدید
            self.move_log.append((self.selected.row, self.selected.col, row, col))  # اضافه کردن حرکت به لاگ
            if len(self.move_log) > 30:
                # اگه لاگ حرکت‌ها بیشتر از 30 تا بشه
                self.move_log.pop(0)  # حذف قدیمی‌ترین حرکت
            kinged = False  # پرچم برای شاه شدن
            if not piece.is_player_2 and row == 0 and not piece.king:
                # اگه مهره بازیکن 1 به ردیف آخر (0) برسه و شاه نباشه
                piece.make_king()  # تبدیل مهره به شاه
                kinged = True  # پرچم شاه شدن فعال می‌شه
                self.no_capture_or_king_moves = 0  # ریست شمارش حرکت‌ها
            elif piece.is_player_2 and row == 7 and not piece.king:
                # اگه مهره بازیکن 2 به ردیف آخر (7) برسه و شاه نباشه
                piece.make_king()  # تبدیل مهره به شاه
                kinged = True  # پرچم شاه شدن فعال می‌شه
                self.no_capture_or_king_moves = 0  # ریست شمارش حرکت‌ها
            if self.interface:
                # اگه رابط کاربری وجود داشته باشه
                self.interface.animate_move(piece, start_row, start_col, row, col)  # اجرای انیمیشن حرکت
            if kinged:
                # اگه مهره شاه شده باشه
                self.multi_jump_active = False  # غیرفعال کردن حالت پرش چندگانه
                self.change_turn()  # عوض کردن نوبت
                return True  # برگردوندن True برای حرکت موفق
            if skipped:
                # اگه پرش انجام شده باشه
                additional_moves = self.get_valid_moves(piece)  # گرفتن پرش‌های بعدی برای همون مهره
                if any(skipped for skipped in additional_moves.values()):
                    # اگه پرش‌های بعدی ممکن باشن
                    self.valid_moves = {move: skip for move, skip in additional_moves.items() if
                                        skip}  # به‌روزرسانی حرکت‌های معتبر فقط با پرش‌ها
                    self.multi_jump_active = True  # فعال کردن حالت پرش چندگانه
                    return True  # برگردوندن True و نگه داشتن نوبت
            self.multi_jump_active = False  # غیرفعال کردن حالت پرش چندگانه
            self.change_turn()  # عوض کردن نوبت
            return True  # برگردوندن True برای حرکت موفق
        return False  # اگه مهره‌ای وجود نداشته باشه، برگردوندن False

    def get_valid_moves(self, piece):
        # محاسبه حرکت‌های معتبر برای یه مهره
        moves = {}  # دیکشنری برای حرکت‌های ساده
        jumps = {}  # دیکشنری برای پرش‌ها
        row, col = piece.row, piece.col  # گرفتن مختصات مهره

        # تعیین جهت‌های مجاز برای مهره
        directions = []
        if not piece.is_player_2 and not piece.king:
            # اگه مهره بازیکن 1 باشه و شاه نباشه
            directions = [(-1, -1), (-1, 1)]  # حرکت به بالا (جهت‌های مورب)
        elif piece.is_player_2 and not piece.king:
            # اگه مهره بازیکن 2 باشه و شاه نباشه
            directions = [(1, -1), (1, 1)]  # حرکت به پایین (جهت‌های مورب)
        elif piece.king:
            # اگه مهره شاه باشه
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # حرکت تو همه جهت‌ها

        has_jump = False  # پرچم برای وجود پرش
        # حلقه برای بررسی همه مهره‌های بازیکن فعلی
        for r in range(BOARD_SIZE):
            # برای هر ردیف
            for c in range(BOARD_SIZE):
                # برای هر ستون
                p = self.board.board[r][c]  # گرفتن مهره تو موقعیت فعلی
                if p and p.is_player_2 == piece.is_player_2:
                    # اگه مهره‌ای وجود داشته باشه و متعلق به بازیکن فعلی باشه
                    # تعیین جهت‌های مجاز برای این مهره
                    piece_directions = [(-1, -1), (-1, 1)] if not p.is_player_2 and not p.king else \
                        [(1, -1), (1, 1)] if p.is_player_2 and not p.king else \
                            [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    # بررسی هر جهت
                    for dr, dc in piece_directions:
                        # برای هر جهت مجاز
                        new_row, new_col = p.row + dr, p.col + dc  # محاسبه خانه مجاور
                        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                            # اگه خانه مجاور تو صفحه باشه
                            target = self.board.board[new_row][new_col]  # گرفتن مهره تو خانه مجاور
                            if target and target.is_player_2 != p.is_player_2:
                                # اگه مهره حریف تو خانه مجاور باشه
                                jump_row, jump_col = new_row + dr, new_col + dc  # محاسبه خانه بعد از پرش
                                if (0 <= jump_row < BOARD_SIZE and 0 <= jump_col < BOARD_SIZE and
                                        not self.board.board[jump_row][jump_col]):
                                    # اگه خانه بعد از پرش تو صفحه باشه و خالی باشه
                                    has_jump = True  # پرچم پرش فعال می‌شه
                                    if p == piece:
                                        # اگه مهره بررسی‌شده همون مهره انتخاب‌شده باشه
                                        jumps[(jump_row, jump_col)] = [target]  # اضافه کردن پرش به دیکشنری پرش‌ها

        if has_jump:
            # اگه حداقل یه پرش برای هر مهره‌ای وجود داشته باشه
            return jumps if jumps else {}  # برگردوندن پرش‌ها (یا خالی اگه مهره انتخاب‌شده پرش نداشته باشه)

        # اگه هیچ پرشی وجود نداشته باشه، حرکت‌های ساده رو محاسبه کن
        for dr, dc in directions:
            # برای هر جهت مجاز
            new_row, new_col = row + dr, col + dc  # محاسبه خانه مجاور
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                # اگه خانه مجاور تو صفحه باشه
                target = self.board.board[new_row][new_col]  # گرفتن مهره تو خانه مجاور
                if not target:
                    # اگه خانه خالی باشه
                    moves[(new_row, new_col)] = []  # اضافه کردن حرکت ساده به دیکشنری

        return moves  # برگردوندن حرکت‌های ساده

    def draw_valid_moves(self):
        # نمایش حرکت‌های معتبر روی صفحه
        if self.valid_moves:
            # اگه حرکت‌های معتبری وجود داشته باشه
            for move in self.valid_moves:
                # برای هر حرکت معتبر
                row, col = move  # گرفتن مختصات مقصد
                # رسم دایره قرمز برای نمایش حرکت معتبر
                pygame.draw.circle(self.screen, RED,
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS), 20)

    def change_turn(self):
        # عوض کردن نوبت بازیکن
        self.selected = None  # ریست مهره انتخاب‌شده
        self.valid_moves = {}  # ریست حرکت‌های معتبر
        self.turn = not self.turn  # تغییر نوبت (از True به False یا برعکس)

    def get_hint(self):
        # گرفتن پیشنهاد حرکت از هوش مصنوعی
        if self.game_over:
            # اگه بازی تموم شده باشه
            return None  # چیزی برنگردون
        ai = self.player_2_ai if self.turn else self.player_1_ai  # انتخاب هوش مصنوعی بر اساس نوبت
        if ai and not (self.player_1_ai if not self.turn else self.player_2_ai):
            # اگه هوش مصنوعی برای بازیکن فعلی وجود داشته باشه و برای حریف نه
            # حلقه برای بررسی همه مهره‌ها
            for row in range(BOARD_SIZE):
                # برای هر ردیف
                for col in range(BOARD_SIZE):
                    # برای هر ستون
                    piece = self.board.board[row][col]  # گرفتن مهره تو موقعیت فعلی
                    if piece and piece.is_player_2 == self.turn:
                        # اگه مهره متعلق به بازیکن فعلی باشه
                        moves = self.get_valid_moves(piece)  # گرفتن حرکت‌های معتبر
                        if moves:
                            # اگه حرکت‌های معتبری وجود داشته باشه
                            # گرفتن حرکت پیشنهادی از هوش مصنوعی
                            move = ai.suggest_move(self, moves) if hasattr(ai, 'suggest_move') else ai.act(self, moves)
                            if move:
                                # اگه حرکتی پیشنهاد شده باشه
                                return (piece, move)  # برگردوندن مهره و حرکت پیشنهادی
        return None  # اگه پیشنهادی نباشه، None برگردون