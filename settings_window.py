import pygame
import os
from .game import Game
from .windows import SettingsWindow, AIProgressWindow, HelpWindow, AboutWindow
from .constants import WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_WIDTH, PANEL_WIDTH, MENU_HEIGHT, BORDER_THICKNESS, \
    SQUARE_SIZE, BUTTON_SPACING_FROM_BOTTOM, ANIMATION_FRAMES, PLAYER_IMAGE_SIZE, BLACK, LIGHT_GRAY, SKY_BLUE, \
    LIGHT_GREEN, BLUE, WHITE, GRAY, LANGUAGES
from .utils import hex_to_rgb
import tkinter as tk
from tkinter import messagebox
import sys
import time
from .config import load_config, save_stats


class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 16)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


class GameInterface:
    def __init__(self, settings):
        self.settings = settings
        # بارگذاری تنظیمات از فایل
        loaded_config = load_config()
        for key, value in loaded_config.items():
            if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                value = hex_to_rgb(value)
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.root = tk.Tk()
        self.root.withdraw()
        self.game = Game(settings, self)
        self.screen = self.game.screen
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
        self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)
        self.last_update = pygame.time.get_ticks()
        self.new_game_button = Button(BOARD_WIDTH + BORDER_THICKNESS * 2 + 20,
                                      WINDOW_HEIGHT - BUTTON_SPACING_FROM_BOTTOM - 100, PANEL_WIDTH - 40, 40,
                                      LANGUAGES[settings.language]["start_game"], SKY_BLUE)
        self.reset_scores_button = Button(BOARD_WIDTH + BORDER_THICKNESS * 2 + 20,
                                          WINDOW_HEIGHT - BUTTON_SPACING_FROM_BOTTOM - 50, PANEL_WIDTH - 40, 40,
                                          LANGUAGES[self.settings.language]["reset_scores"], LIGHT_GREEN)
        self.player_1_move_ready = True
        self.player_2_move_ready = True
        self.player_1_pause_start = None
        self.player_2_pause_start = None
        self.pending_settings = None
        self.current_hand = 0
        self.pause_start_time = None
        self.auto_start_triggered = False
        self.player_1_image_surface = None
        self.player_2_image_surface = None
        self.player_1_piece_surface = None
        self.player_1_king_surface = None
        self.player_2_piece_surface = None
        self.player_2_king_surface = None
        self.load_player_images()
        self.load_piece_images()
        self.ai_move_start_time = None

    def apply_pending_settings(self):
        if self.pending_settings:
            # زبان فقط با ری‌استارت اعمال می‌شود، بنابراین آن را نادیده می‌گیریم
            self.settings.player_starts = self.pending_settings["player_starts"]
            self.settings.use_timer = self.pending_settings["use_timer"]
            self.settings.game_time = self.pending_settings["game_time"]
            self.settings.player_1_color = hex_to_rgb(self.pending_settings["player_1_color"])
            self.settings.player_2_color = hex_to_rgb(self.pending_settings["player_2_color"])
            self.settings.board_color_1 = hex_to_rgb(self.pending_settings["board_color_1"])
            self.settings.board_color_2 = hex_to_rgb(self.pending_settings["board_color_2"])
            self.settings.piece_style = self.pending_settings.get("piece_style", self.settings.piece_style)
            self.settings.sound_enabled = self.pending_settings["sound_enabled"]
            self.settings.ai_pause_time = self.pending_settings.get("ai_pause_time", self.settings.ai_pause_time)
            self.settings.ai_ability = self.pending_settings["ai_ability"]
            self.settings.game_mode = self.pending_settings.get("game_mode", self.settings.game_mode)
            self.settings.ai_vs_ai_mode = self.pending_settings.get("ai_vs_ai_mode", self.settings.ai_vs_ai_mode)
            self.settings.repeat_hands = self.pending_settings.get("repeat_hands", self.settings.repeat_hands)
            self.settings.player_1_name = self.pending_settings.get("player_1_name", self.settings.player_1_name)
            self.settings.player_2_name = self.pending_settings.get("player_2_name", self.settings.player_2_name)
            self.settings.ai_1_name = self.pending_settings.get("ai_1_name", self.settings.ai_1_name)
            self.settings.ai_2_name = self.pending_settings.get("ai_2_name", self.settings.ai_2_name)
            self.settings.player_1_image = self.pending_settings.get("player_1_image", self.settings.player_1_image)
            self.settings.player_2_image = self.pending_settings.get("player_2_image", self.settings.player_2_image)
            self.settings.ai_1_image = self.pending_settings.get("ai_1_image", self.settings.ai_1_image)
            self.settings.ai_2_image = self.pending_settings.get("ai_2_image", self.settings.ai_2_image)
            self.settings.player_1_piece_image = self.pending_settings.get("player_1_piece_image",
                                                                           self.settings.player_1_piece_image)
            self.settings.player_1_king_image = self.pending_settings.get("player_1_king_image",
                                                                          self.settings.player_1_king_image)
            self.settings.player_2_piece_image = self.pending_settings.get("player_2_piece_image",
                                                                           self.settings.player_2_piece_image)
            self.settings.player_2_king_image = self.pending_settings.get("player_2_king_image",
                                                                          self.settings.player_2_king_image)
            self.settings.pause_between_hands = self.pending_settings.get("pause_between_hands",
                                                                          self.settings.pause_between_hands)
            # اعمال تنظیمات جدید نوع AI
            self.settings.player_1_ai_type = self.pending_settings["player_1_ai_type"]
            self.settings.player_2_ai_type = self.pending_settings["player_2_ai_type"]
            self.pending_settings = None
            self.load_player_images()
            self.load_piece_images()
            self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
            self.reset_scores_button.text = LANGUAGES[self.settings.language]["reset_scores"]
            self.game = Game(self.settings, self)
            self.screen = self.game.screen
            self.current_hand = 0
            self.auto_start_triggered = False

    def load_player_images(self):
        player_1_image_path = self.settings.player_1_image if self.settings.game_mode in ["human_vs_human",
                                                                                          "human_vs_ai"] else self.settings.ai_1_image
        player_2_image_path = self.settings.player_2_image if self.settings.game_mode == "human_vs_human" else self.settings.ai_2_image

        for path, attr in [(player_1_image_path, 'player_1_image_surface'),
                           (player_2_image_path, 'player_2_image_surface')]:
            surface = None
            if path and os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    image = pygame.image.load(path)
                    if image.get_size() != (0, 0):  # چک کردن تصویر نامعتبر
                        image = pygame.transform.scale(image, (PLAYER_IMAGE_SIZE, PLAYER_IMAGE_SIZE))
                        mask = pygame.Surface((PLAYER_IMAGE_SIZE, PLAYER_IMAGE_SIZE), pygame.SRCALPHA)
                        pygame.draw.circle(mask, (255, 255, 255, 255), (PLAYER_IMAGE_SIZE // 2, PLAYER_IMAGE_SIZE // 2),
                                           PLAYER_IMAGE_SIZE // 2)
                        image.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
                        surface = image
                    else:
                        print(f"Invalid image file {path}: Empty or corrupted")
                except pygame.error as e:
                    print(f"Error loading image {path}: {e}")
            if surface is None:
                print(f"Using default image for {attr}")
            setattr(self, attr, surface)

    def load_piece_images(self):
        piece_size = SQUARE_SIZE - 20
        for path, attr in [
            (self.settings.player_1_piece_image, 'player_1_piece_surface'),
            (self.settings.player_1_king_image, 'player_1_king_surface'),
            (self.settings.player_2_piece_image, 'player_2_piece_surface'),
            (self.settings.player_2_king_image, 'player_2_king_surface')
        ]:
            surface = None
            if path and os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    image = pygame.image.load(path)
                    if image.get_size() != (0, 0):
                        image = pygame.transform.scale(image, (piece_size, piece_size))
                        mask = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                        pygame.draw.circle(mask, (255, 255, 255, 255), (piece_size // 2, piece_size // 2),
                                           piece_size // 2)
                        image.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
                        surface = image
                    else:
                        print(f"Invalid image file {path}: Empty or corrupted")
                except pygame.error as e:
                    print(f"Failed to load image {path}: {e}")
            if surface is None:
                print(f"Using default piece style for {attr}")
            setattr(self, attr, surface)

    def draw_piece(self, screen, piece, x=None, y=None, animate=False):
        draw_x = x if x is not None else piece.x
        draw_y = y if y is not None else piece.y
        radius = SQUARE_SIZE // 2 - 10
        color = self.settings.player_2_color if piece.is_player_2 else self.settings.player_1_color

        piece_surface = None
        if piece.is_player_2:
            piece_surface = self.player_2_king_surface if piece.king else self.player_2_piece_surface
        else:
            piece_surface = self.player_1_king_surface if piece.king else self.player_1_piece_surface

        if piece_surface is not None:
            screen.blit(piece_surface, (draw_x - radius, draw_y - radius))
        else:
            if self.settings.piece_style == "circle":
                pygame.draw.circle(screen, color, (draw_x, draw_y), radius)
            elif self.settings.piece_style == "outlined_circle":
                pygame.draw.circle(screen, color, (draw_x, draw_y), radius)
                pygame.draw.circle(screen, BLACK, (draw_x, draw_y), radius, 2)
            elif self.settings.piece_style == "square":
                pygame.draw.rect(screen, color, (draw_x - radius // 2, draw_y - radius // 2, radius, radius))
            if piece.king:
                pygame.draw.circle(screen, GRAY, (draw_x, draw_y), radius - 10)

    def animate_move(self, piece, start_row, start_col, end_row, end_col):
        if start_row is None or start_col is None or end_row is None or end_col is None:
            return
        start_x = start_col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS
        start_y = start_row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS
        end_x = end_col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS
        end_y = end_row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS
        radius = SQUARE_SIZE // 2 - 10

        piece_surface = None
        if piece.is_player_2:
            piece_surface = self.player_2_king_surface if piece.king else self.player_2_piece_surface
        else:
            piece_surface = self.player_1_king_surface if piece.king else self.player_1_piece_surface

        for frame in range(ANIMATION_FRAMES + 1):
            t = frame / ANIMATION_FRAMES
            current_x = start_x + (end_x - start_x) * t
            current_y = start_y + (end_y - start_y) * t
            self.draw_game()
            if piece_surface is not None:
                self.screen.blit(piece_surface, (current_x - radius, current_y - radius))
            else:
                self.draw_piece(self.screen, piece, current_x, current_y)
            self.game.draw_valid_moves()
            pygame.display.update()
            pygame.time.wait(25)

    def draw_default_image(self, name, x, y):
        surface = pygame.Surface((PLAYER_IMAGE_SIZE, PLAYER_IMAGE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(surface, LIGHT_GRAY, (PLAYER_IMAGE_SIZE // 2, PLAYER_IMAGE_SIZE // 2),
                           PLAYER_IMAGE_SIZE // 2)
        text = self.small_font.render(name[:2], True, BLACK)
        text_rect = text.get_rect(center=(PLAYER_IMAGE_SIZE // 2, PLAYER_IMAGE_SIZE // 2))
        surface.blit(text, text_rect)
        self.screen.blit(surface, (x, y))

    def draw_game(self):
        self.screen.fill(self.settings.board_color_1)
        pygame.draw.rect(self.screen, LIGHT_GRAY, (0, 0, WINDOW_WIDTH, MENU_HEIGHT))

        ai_progress_text = self.small_font.render(LANGUAGES[self.settings.language]["ai_progress"], True, BLACK)
        self.screen.blit(ai_progress_text, (10, 5))
        az_progress_text = self.small_font.render(LANGUAGES[self.settings.language]["az_progress"], True, BLACK)
        self.screen.blit(az_progress_text, (130, 5))
        settings_text = self.small_font.render(LANGUAGES[self.settings.language]["settings"], True, BLACK)
        self.screen.blit(settings_text, (260, 5))
        help_text = self.small_font.render(LANGUAGES[self.settings.language]["help"], True, BLACK)
        self.screen.blit(help_text, (355, 5))
        about_text = self.small_font.render(LANGUAGES[self.settings.language]["about_me"], True, BLACK)
        self.screen.blit(about_text, (420, 5))
        pygame.draw.rect(self.screen, BLUE, (0, MENU_HEIGHT, BOARD_WIDTH + BORDER_THICKNESS * 2,
                                             BOARD_WIDTH + BORDER_THICKNESS * 3), BORDER_THICKNESS)
        self.game.board.draw(self.screen, self.settings.board_color_1, self.settings.board_color_2)
        for row in self.game.board.board:
            for piece in row:
                if piece:
                    self.draw_piece(self.screen, piece)
        self.game.draw_valid_moves()
        self.draw_side_panel()
        pygame.draw.rect(self.screen, LIGHT_GRAY, (0, WINDOW_HEIGHT - 30, WINDOW_WIDTH, 30))
        footer_text = self.small_font.render(LANGUAGES[self.settings.language]["footer"], True, BLACK)
        footer_rect = footer_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 15))
        self.screen.blit(footer_text, footer_rect)

    def draw_side_panel(self):
        panel_rect = pygame.Rect(BOARD_WIDTH + BORDER_THICKNESS * 2, MENU_HEIGHT, PANEL_WIDTH, WINDOW_HEIGHT - MENU_HEIGHT - 30)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        y = MENU_HEIGHT + 20
        result_title = self.font.render(LANGUAGES[self.settings.language]["game_result"], True, BLACK)
        result_rect = result_title.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH // 2, y))
        self.screen.blit(result_title, result_rect)
        y += 40
        if self.game.game_over:
            if self.game.winner is None:
                result_text = "بازی مساوی شد" if self.settings.language == "fa" else "Game is a Draw"
            elif not self.game.winner:
                result_text = LANGUAGES[self.settings.language]["player_wins"]
            else:
                result_text = LANGUAGES[self.settings.language]["ai_wins"]
            result_display = self.small_font.render(result_text, True, BLACK)
            result_display_rect = result_display.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH // 2, y))
            self.screen.blit(result_display, result_display_rect)
        y += 80

        player_1_name = self.settings.player_1_name if self.settings.game_mode in ["human_vs_human", "human_vs_ai"] else self.settings.ai_1_name
        player_2_name = self.settings.player_2_name if self.settings.game_mode == "human_vs_human" else self.settings.ai_2_name

        if self.player_1_image_surface:
            self.screen.blit(self.player_1_image_surface, (BOARD_WIDTH + BORDER_THICKNESS * 2 + 20, y))
        else:
            self.draw_default_image(player_1_name, BOARD_WIDTH + BORDER_THICKNESS * 2 + 20, y)

        vs_font = pygame.font.SysFont('Arial', 24, bold=True)
        vs_text = vs_font.render("vs", True, BLUE)
        vs_text.set_alpha(200)
        vs_rect = vs_text.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH // 2, y + PLAYER_IMAGE_SIZE // 2))
        pygame.draw.rect(self.screen, LIGHT_GRAY, (vs_rect.x - 5, vs_rect.y - 5, vs_rect.width + 10, vs_rect.height + 10), border_radius=5)
        self.screen.blit(vs_text, vs_rect)

        if self.player_2_image_surface:
            self.screen.blit(self.player_2_image_surface, (BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH - 20 - PLAYER_IMAGE_SIZE, y))
        else:
            self.draw_default_image(player_2_name, BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH - 20 - PLAYER_IMAGE_SIZE, y)

        y += 15

        player_1_name_text = self.small_font.render(player_1_name, True, BLACK)
        player_1_name_rect = player_1_name_text.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + 20 + PLAYER_IMAGE_SIZE // 2, y + PLAYER_IMAGE_SIZE + 10))
        self.screen.blit(player_1_name_text, player_1_name_rect)

        player_2_name_text = self.small_font.render(player_2_name, True, BLACK)
        player_2_name_rect = player_2_name_text.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH - 20 - PLAYER_IMAGE_SIZE // 2, y + PLAYER_IMAGE_SIZE + 10))
        self.screen.blit(player_2_name_text, player_2_name_rect)

        y += PLAYER_IMAGE_SIZE + 50

        # جدول اسکوربرد با گرادیانت و سایه
        table_rect = pygame.Rect(BOARD_WIDTH + BORDER_THICKNESS * 2 + 20, y, PANEL_WIDTH - 40, 80)
        shadow_rect = pygame.Rect(table_rect.x + 5, table_rect.y + 5, table_rect.width, table_rect.height)
        pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=10)

        gradient_surface = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
        for i in range(table_rect.height):
            color = (255 - i * 2, 255 - i * 2, 255 - i * 2)
            pygame.draw.line(gradient_surface, color, (0, i), (table_rect.width, i))
        mask = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, table_rect.width, table_rect.height), border_radius=10)
        gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
        self.screen.blit(gradient_surface, (table_rect.x, table_rect.y))

        pygame.draw.rect(self.screen, BLACK, table_rect, 2, border_radius=10)
        pygame.draw.line(self.screen, BLACK, (table_rect.x + table_rect.width // 3, table_rect.y),
                         (table_rect.x + table_rect.width // 3, table_rect.y + table_rect.height), 2)
        pygame.draw.line(self.screen, BLACK, (table_rect.x + 2 * table_rect.width // 3, table_rect.y),
                         (table_rect.x + 2 * table_rect.width // 3, table_rect.y + table_rect.height), 2)
        pygame.draw.line(self.screen, BLACK, (table_rect.x, table_rect.y + table_rect.height // 2),
                         (table_rect.x + table_rect.width, table_rect.y + table_rect.height // 2), 2)

        wins_1 = self.small_font.render(str(self.game.player_1_wins), True, BLACK)
        wins_1_rect = wins_1.get_rect(center=(table_rect.x + table_rect.width // 6, table_rect.y + table_rect.height // 4))
        self.screen.blit(wins_1, wins_1_rect)

        wins_text = self.small_font.render(LANGUAGES[self.settings.language]["wins"], True, BLACK)
        wins_text_rect = wins_text.get_rect(center=(table_rect.x + table_rect.width // 2, table_rect.y + table_rect.height // 4))
        self.screen.blit(wins_text, wins_text_rect)

        wins_2 = self.small_font.render(str(self.game.player_2_wins), True, BLACK)
        wins_2_rect = wins_2.get_rect(center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + table_rect.height // 4))
        self.screen.blit(wins_2, wins_2_rect)

        pieces_1 = self.small_font.render(str(self.game.board.player_1_left), True, BLACK)
        pieces_1_rect = pieces_1.get_rect(center=(table_rect.x + table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4))
        self.screen.blit(pieces_1, pieces_1_rect)

        pieces_text = self.small_font.render(LANGUAGES[self.settings.language]["pieces"], True, BLACK)
        pieces_text_rect = pieces_text.get_rect(center=(table_rect.x + table_rect.width // 2, table_rect.y + 3 * table_rect.height // 4))
        self.screen.blit(pieces_text, pieces_text_rect)

        pieces_2 = self.small_font.render(str(self.game.board.player_2_left), True, BLACK)
        pieces_2_rect = pieces_2.get_rect(center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4))
        self.screen.blit(pieces_2, pieces_2_rect)

        y += 100

        # جدول تایمر (همیشه نمایش داده می‌شود)
        time_title = self.small_font.render(LANGUAGES[self.settings.language]["time"], True, BLACK)
        time_title_rect = time_title.get_rect(center=(BOARD_WIDTH + BORDER_THICKNESS * 2 + PANEL_WIDTH // 2, y))
        self.screen.blit(time_title, time_title_rect)
        y += 20

        timer_rect = pygame.Rect(BOARD_WIDTH + BORDER_THICKNESS * 2 + 20, y, PANEL_WIDTH - 40, 40)
        shadow_rect = pygame.Rect(timer_rect.x + 5, timer_rect.y + 5, timer_rect.width, timer_rect.height)
        pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=10)
        gradient_surface = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
        for i in range(timer_rect.height):
            color = (135 - i * 2, 206 - i * 2, 235 - i * 2)
            pygame.draw.line(gradient_surface, color, (0, i), (timer_rect.width, i))
        mask = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, timer_rect.width, timer_rect.height), border_radius=10)
        gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
        self.screen.blit(gradient_surface, (timer_rect.x, timer_rect.y))

        pygame.draw.rect(self.screen, BLACK, timer_rect, 2, border_radius=10)
        pygame.draw.line(self.screen, BLACK, (timer_rect.x + timer_rect.width // 2, timer_rect.y),
                         (timer_rect.x + timer_rect.width // 2, timer_rect.y + timer_rect.height), 2)

        bold_font = pygame.font.SysFont('Arial', 18, bold=True)
        player_1_time = self.game.timer.get_current_time(False)
        player_2_time = self.game.timer.get_current_time(True)
        timer_1 = bold_font.render(f"{int(player_1_time)} s", True, BLACK if not self.game.turn else GRAY)
        timer_1_rect = timer_1.get_rect(center=(timer_rect.x + timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
        self.screen.blit(timer_1, timer_1_rect)

        timer_2 = bold_font.render(f"{int(player_2_time)} s", True, BLACK if self.game.turn else GRAY)
        timer_2_rect = timer_2.get_rect(center=(timer_rect.x + 3 * timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
        self.screen.blit(timer_2, timer_2_rect)

        y += 50

        self.new_game_button.draw(self.screen)
        self.reset_scores_button.draw(self.screen)

    def _handle_events(self, settings_window, ai_progress_window, az_progress_window, help_window, about_window):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # مدیریت رویداد برای منوی بالا
                if pos[1] < MENU_HEIGHT:
                    if not (settings_window.is_open or ai_progress_window.is_open or az_progress_window.is_open or help_window.is_open or about_window.is_open):
                        if pos[0] < 130:
                            ai_progress_window.create_widgets()
                            ai_progress_window.is_open = True
                        elif 140 <= pos[0] < 260:
                            az_progress_window.create_widgets()
                            az_progress_window.is_open = True
                        elif 270 <= pos[0] < 355:
                            settings_window.create_widgets()
                            settings_window.is_open = True
                        elif 365 <= pos[0] < 410:
                            help_window.create_widgets()
                            help_window.is_open = True
                        elif 420 <= pos[0] < 480:
                            about_window.create_widgets()
                            about_window.is_open = True

                # مدیریت دکمه شروع یا بازی جدید
                elif self.new_game_button.is_clicked(pos):
                    if not self.game.game_started:
                        # شروع بازی جدید (حتی اگر بازی قبلی تمام شده باشد)
                        self.game.start_game()
                        self.new_game_button.text = LANGUAGES[self.settings.language]["new_game"]  # تبدیل به "NEW GAME"
                        self.last_update = pygame.time.get_ticks()
                        self.current_hand = 0  # ریست تعداد دست‌ها
                        self.auto_start_triggered = False

                        # پخش صدای شروع بازی
                        if self.settings.sound_enabled and os.path.exists('start.wav') and os.path.getsize('start.wav') > 0:
                            try:
                                pygame.mixer.Sound('start.wav').play()
                            except pygame.error as e:
                                print(f"Error playing start.wav: {e}")

                    elif self.game.game_started and not self.game.game_over:
                        # نمایش اخطار فقط اگر بازی در جریان است و پایان نیافته
                        confirmed = messagebox.askyesno(
                            LANGUAGES[self.settings.language]["warning"],
                            LANGUAGES[self.settings.language]["new_game_warning"],
                            parent=self.root
                        )
                        if confirmed:
                            # بازچینی و ریست بازی
                            self.game.game_over = True
                            self.game.score_updated = False
                            self.game.init_game()
                            self.game.game_started = False
                            self.game.game_over = False
                            self.game.winner = None
                            self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]  # تبدیل به "START GAME"
                            self.last_update = pygame.time.get_ticks()
                            self.current_hand = 0
                            self.auto_start_triggered = False
                            pygame.event.clear()

                    elif self.game.game_over:
                        # اگر بازی تمام شده باشد، مستقیماً به START GAME برگردد
                        self.game.game_over = True
                        self.game.score_updated = False
                        self.game.init_game()
                        self.game.game_started = False
                        self.game.game_over = False
                        self.game.winner = None
                        self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]  # تبدیل به "START GAME"
                        self.last_update = pygame.time.get_ticks()
                        self.current_hand = 0
                        self.auto_start_triggered = False
                        pygame.event.clear()

                # مدیریت دکمه ریست امتیازات
                elif self.reset_scores_button.is_clicked(pos):
                    if messagebox.askyesno(
                        LANGUAGES[self.settings.language]["warning"],
                        LANGUAGES[self.settings.language]["reset_scores_warning"],
                        parent=self.root
                    ):
                        self.game.player_1_wins = 0
                        self.game.player_2_wins = 0
                        save_stats({"player_1_wins": 0, "player_2_wins": 0})

                # مدیریت کلیک‌های مربوط به بازی
                elif self.game.game_started:
                    self.game.handle_click(pos)

        return True

    def update(self):
        current_time = pygame.time.get_ticks()
        self.game._update_game()
        if self.game.game_started and not self.game.game_over:
            if self.settings.game_mode == "ai_vs_ai":
                if self.player_1_move_ready and not self.game.turn:
                    self.player_1_move_ready = False
                    self.ai_move_start_time = current_time
                    self.game.make_ai_move(self.game.player_1_ai)
                elif self.player_2_move_ready and self.game.turn:
                    self.player_2_move_ready = False
                    self.ai_move_start_time = current_time
                    self.game.make_ai_move(self.game.player_2_ai)
                elif not self.player_1_move_ready and not self.game.turn:
                    if current_time - self.ai_move_start_time >= self.settings.ai_pause_time:
                        self.player_1_move_ready = True
                elif not self.player_2_move_ready and self.game.turn:
                    if current_time - self.ai_move_start_time >= self.settings.ai_pause_time:
                        self.player_2_move_ready = True
            elif self.settings.game_mode == "human_vs_ai" and self.game.turn:
                if self.player_2_move_ready:
                    self.player_2_move_ready = False
                    self.ai_move_start_time = current_time
                    self.game.make_ai_move(self.game.player_2_ai)
                elif current_time - self.ai_move_start_time >= self.settings.ai_pause_time:
                    self.player_2_move_ready = True

        if self.game.game_over and self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game":
            if self.current_hand < self.settings.repeat_hands:
                if self.pause_start_time is None:
                    self.pause_start_time = current_time
                elif current_time - self.pause_start_time >= self.settings.pause_between_hands:
                    self.game.reset_board()  # ریست کردن صفحه برای دست جدید
                    self.game.start_game()   # شروع دست جدید
                    self.current_hand += 1
                    self.pause_start_time = None
                    self.auto_start_triggered = False
            else:
                self.auto_start_triggered = True
                self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                self.game.game_started = False

        self.draw_game()
        pygame.display.update()

    def close_windows(self, settings_window, ai_progress_window, az_progress_window, help_window, about_window):
        """بستن تمام پنجره‌های باز و ریشه‌ی اصلی"""
        for window in [settings_window, ai_progress_window, az_progress_window, help_window, about_window]:
            if window.is_open:
                window.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        settings_window = SettingsWindow(self, self.root)
        ai_progress_window = AIProgressWindow(self, self.root)
        az_progress_window = AIProgressWindow(self, self.root)
        help_window = HelpWindow(self, self.root)
        about_window = AboutWindow(self, self.root)

        running = True
        while running:
            running = self._handle_events(settings_window, ai_progress_window, az_progress_window, help_window, about_window)
            for window in [settings_window, ai_progress_window, az_progress_window, help_window, about_window]:
                if window.is_open:
                    window.update()
            self.update()
            try:
                self.root.update()  # آپدیت حلقه‌ی رویداد tkinter
            except tk.TclError:
                running = False

        self.close_windows(settings_window, ai_progress_window, az_progress_window, help_window, about_window)
        pygame.quit()
        sys.exit()