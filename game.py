import pygame
import os
import random
from .board import Board
from .timer import TimerManager
from .ai import AdvancedAI
from .rewards import RewardCalculator
from .alphazero_ai import AlphaZeroAI
from .config import load_config, load_stats, save_stats
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
        self.multi_jump_active = False
        self.reward_calculator = RewardCalculator(self)
        self.update_ai_players()
        self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()
        self.init_game()

    def update_ai_players(self):
        config = load_config()
        if self.settings.game_mode == "human_vs_human":
            self.settings.player_1_ai_type = "none"
            self.settings.player_2_ai_type = "none"
        elif self.settings.game_mode == "human_vs_ai":
            self.settings.player_1_ai_type = "none"
            if self.settings.player_2_ai_type == "none":
                self.settings.player_2_ai_type = "alphazero"
        elif self.settings.game_mode == "ai_vs_ai":
            if self.settings.player_1_ai_type == "none":
                self.settings.player_1_ai_type = "alphazero"
            if self.settings.player_2_ai_type == "none":
                self.settings.player_2_ai_type = "alphazero"

        if self.settings.player_1_ai_type == "advanced":
            self.player_1_ai = AdvancedAI(
                player_color=self.settings.player_1_color,
                ability=config.get("ai_1_ability", 5),
                model_name="ai_1_model",
                ai_id="ai_1"
            )
        elif self.settings.player_1_ai_type == "alphazero":
            self.player_1_ai = AlphaZeroAI(
                game=self,
                color=self.settings.player_1_color,
                model_name="az_1_model",
                az_id="az1"
            )
        else:
            self.player_1_ai = None

        if self.settings.player_2_ai_type == "advanced":
            self.player_2_ai = AdvancedAI(
                player_color=self.settings.player_2_color,
                ability=config.get("ai_2_ability", 5),
                model_name="ai_2_model",
                ai_id="ai_2"
            )
        elif self.settings.player_2_ai_type == "alphazero":
            self.player_2_ai = AlphaZeroAI(
                game=self,
                color=self.settings.player_2_color,
                model_name="az_2_model",
                az_id="az2"
            )
        else:
            self.player_2_ai = None

    def init_game(self):
        self.board = Board(self.settings)
        self.selected = None
        self.turn = False if self.settings.player_starts else True
        self.valid_moves = {}
        self.game_over = False
        self.winner = None
        self.last_state = None
        self.last_action = None
        self.move_log = []
        self.click_log = []
        self.game_started = False
        self.score_updated = False
        self.no_capture_or_king_moves = 0
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.repeat_count = 0

    def start_game(self):
        self.game_started = True
        self.game_over = False
        self.winner = None
        self.selected = None
        self.valid_moves = {}
        self.multi_jump_active = False
        self.last_update_time = pygame.time.get_ticks()
        self.timer.start_game()
        self.update_ai_players()
        if self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game":
            self.repeat_count = self.settings.repeat_hands
        else:
            self.repeat_count = 1

    def reset_board(self):
        self.board = Board(self.settings)
        self.selected = None
        self.turn = False if self.settings.player_starts else True
        self.valid_moves = {}
        self.game_over = False
        self.winner = None
        self.move_log = []
        self.click_log = []
        self.no_capture_or_king_moves = 0
        self.score_updated = False
        self.multi_jump_active = False
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.update_ai_players()

    def _update_game(self):
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        if not self.game_started or self.game_over:
            return
        player_1_moves = False
        player_2_moves = False
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.board[row][col]
                if piece:
                    moves = self.get_valid_moves(piece)
                    if moves:
                        if not piece.is_player_2:
                            player_1_moves = True
                        else:
                            player_2_moves = True
        if not player_1_moves:
            self.winner = True
            self.game_over = True
        elif not player_2_moves:
            self.winner = False
            self.game_over = True
        elif self.no_capture_or_king_moves >= 40:
            self.winner = None
            self.game_over = True
        if self.settings.use_timer:
            player_1_time = self.timer.get_current_time(False)
            player_2_time = self.timer.get_current_time(True)
            if player_1_time <= 0:
                self.winner = True
                self.game_over = True
                self.time_up = True
            elif player_2_time <= 0:
                self.winner = False
                self.game_over = True
                self.time_up = True
        if self.game_over and not self.score_updated:
            self.score_updated = True
            if self.winner is not None:
                if not self.winner:
                    self.player_1_wins += 1
                else:
                    self.player_2_wins += 1
            save_stats({"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins})

    def make_ai_move(self, ai, current_state=None):
        if not ai:
            return
        valid_pieces = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.board[row][col]
                if piece and piece.is_player_2 == self.turn:
                    moves = self.get_valid_moves(piece)
                    if moves:
                        valid_pieces.append((piece, moves))
        if valid_pieces:
            piece, moves = random.choice(valid_pieces)
            valid_moves = moves
            move = ai.act(self, valid_moves)
            if move and piece and move in valid_moves:
                self.last_state = current_state
                self.last_action = move
                self.selected = piece
                self.valid_moves = valid_moves
                if self._move(move[0], move[1]):
                    self.move_log.append((piece.row, piece.col, move[0], move[1]))
                    if len(self.move_log) > 30:
                        self.move_log.pop(0)
                    ai.replay()
                if self.game_over:
                    if isinstance(ai, AdvancedAI):
                        ai.update_target_network()
                    ai.save_model()
                if self.settings.sound_enabled and os.path.exists('move.wav') and os.path.getsize('move.wav') > 0:
                    try:
                        pygame.mixer.Sound('move.wav').play()
                    except pygame.error as e:
                        print(f"Error playing move.wav: {e}")

    def get_reward(self):
        return self.reward_calculator.get_reward()

    def handle_click(self, pos):
        x, y = pos
        self.click_log.append((x, y))
        if len(self.click_log) > 50:
            self.click_log.pop(0)
        if x < BOARD_WIDTH and self.game_started and not self.game_over:
            if self.settings.game_mode == "human_vs_human" or (
                    self.settings.game_mode == "human_vs_ai" and not self.turn):
                row = (y - MENU_HEIGHT - BORDER_THICKNESS) // SQUARE_SIZE
                col = (x - BORDER_THICKNESS) // SQUARE_SIZE
                if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    return self._move(row, col)
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    self.valid_moves = self.get_valid_moves(self.selected)
                return result
        if self.game_over:
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
        return False

    def select(self, row, col):
        if self.game_over or not self.game_started:
            return False
        if self.selected:
            result = self._move(row, col)
            if not result:
                if not self.multi_jump_active:
                    self.selected = None
                    return self.select(row, col)
                return False
        piece = self.board.board[row][col]
        if piece and (piece.is_player_2 == self.turn):
            if self.multi_jump_active and piece != self.selected:
                return False
            self.selected = piece
            self.valid_moves = self.get_valid_moves(piece)
            return True
        return False

    def _move(self, row, col):
        if not self.selected or (row, col) not in self.valid_moves:
            return False
        piece = self.board.board[self.selected.row][self.selected.col]
        if piece:
            start_row, start_col = self.selected.row, self.selected.col
            self.board.board[self.selected.row][self.selected.col] = None
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
                self.no_capture_or_king_moves = 0
            else:
                self.no_capture_or_king_moves += 1
            piece.move(row, col)
            self.board.board[row][col] = piece
            self.move_log.append((self.selected.row, self.selected.col, row, col))
            if len(self.move_log) > 30:
                self.move_log.pop(0)
            kinged = False
            if not piece.is_player_2 and row == 0 and not piece.king:
                piece.make_king()
                kinged = True
                self.no_capture_or_king_moves = 0
            elif piece.is_player_2 and row == 7 and not piece.king:
                piece.make_king()
                kinged = True
                self.no_capture_or_king_moves = 0
            if self.interface:
                self.interface.animate_move(piece, start_row, start_col, row, col)
            if kinged:
                self.multi_jump_active = False
                self.change_turn()
                return True
            if skipped:
                additional_moves = self.get_valid_moves(piece)
                if any(skipped for skipped in additional_moves.values()):
                    self.valid_moves = {move: skip for move, skip in additional_moves.items() if skip}
                    self.multi_jump_active = True
                    return True
            self.multi_jump_active = False
            self.change_turn()
            return True
        return False

    def get_valid_moves(self, piece):
        moves = {}
        jumps = {}
        row, col = piece.row, piece.col
        directions = []
        if not piece.is_player_2 and not piece.king:
            directions = [(-1, -1), (-1, 1)]
        elif piece.is_player_2 and not piece.king:
            directions = [(1, -1), (1, 1)]
        elif piece.king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        has_jump = False
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = self.board.board[r][c]
                if p and p.is_player_2 == piece.is_player_2:
                    piece_directions = [(-1, -1), (-1, 1)] if not p.is_player_2 and not p.king else \
                                    [(1, -1), (1, 1)] if p.is_player_2 and not p.king else \
                                    [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in piece_directions:
                        new_row, new_col = p.row + dr, p.col + dc
                        if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                            target = self.board.board[new_row][new_col]
                            if target and target.is_player_2 != p.is_player_2:
                                jump_row, jump_col = new_row + dr, new_col + dc
                                if (0 <= jump_row < BOARD_SIZE and 0 <= jump_col < BOARD_SIZE and
                                    not self.board.board[jump_row][jump_col]):
                                    has_jump = True
                                    if p == piece:
                                        jumps[(jump_row, jump_col)] = [target]
        if has_jump:
            return jumps if jumps else {}
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                target = self.board.board[new_row][new_col]
                if not target:
                    moves[(new_row, new_col)] = []
        return moves

    def draw_valid_moves(self):
        if self.valid_moves:
            for move in self.valid_moves:
                row, col = move
                pygame.draw.circle(self.screen, RED,
                                  (col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS,
                                   row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS), 20)

    def change_turn(self):
        self.selected = None
        self.valid_moves = {}
        self.turn = not self.turn

    def get_hint(self):
        if self.game_over:
            return None
        ai = self.player_2_ai if self.turn else self.player_1_ai
        if ai and not (self.player_1_ai if not self.turn else self.player_2_ai):
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    piece = self.board.board[row][col]
                    if piece and piece.is_player_2 == self.turn:
                        moves = self.get_valid_moves(piece)
                        if moves:
                            move = ai.suggest_move(self, moves) if hasattr(ai, 'suggest_move') else ai.act(self, moves)
                            if move:
                                return (piece, move)
        return None