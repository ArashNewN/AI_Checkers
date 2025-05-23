import pygame
import numpy as np
import tkinter as tk
from tkinter import messagebox
import sys
import logging

from .game import Game
from .windows import SettingsWindow, AIProgressWindow, HelpWindow, AboutWindow
from .config import load_config, save_stats, log_to_json, _config_manager
from .constants import LANGUAGES
from .utils import hex_to_rgb

# تنظیم لاگینگ
config = _config_manager.load_config()
project_root = _config_manager.get_project_root()
log_file = project_root / "logs" / "interface.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.get("logging_level", "ERROR")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # نمایش در ترمینال
        logging.FileHandler(log_file, mode="a", encoding="utf-8")  # نوشتن در فایل
    ]
)
logger = logging.getLogger(__name__)

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 16)
        self.enabled = True
        log_to_json(
            f"Button initialized: {text}",
            level="DEBUG",
            extra_data={"x": x, "y": y, "width": width, "height": height}
        )

    def draw(self, screen):
        try:
            color = self.color if self.enabled else (self.color[0] // 2, self.color[1] // 2, self.color[2] // 2)
            pygame.draw.rect(screen, color, self.rect, border_radius=5)
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 2, border_radius=5)
            text_surface = self.font.render(self.text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
            log_to_json(
                f"Drew button: {self.text}",
                level="DEBUG",
                extra_data={"enabled": self.enabled, "rect": [self.rect.x, self.rect.y, self.rect.width, self.rect.height]}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing button {self.text}: {str(e)}",
                level="ERROR",
                extra_data={"rect": [self.rect.x, self.rect.y, self.rect.width, self.rect.height]}
            )

    def is_clicked(self, pos):
        clicked = self.rect.collidepoint(pos)
        log_to_json(
            f"Button {self.text} checked for click at {pos}: {'clicked' if clicked else 'not clicked'}",
            level="DEBUG",
            extra_data={"pos": list(pos)}
        )
        return clicked


class GameInterface:
    def __init__(self, settings):
        try:
            self.settings = settings
            self.config = load_config()
            self.WINDOW_WIDTH = self.config['window_width']
            self.WINDOW_HEIGHT = self.config['window_height']
            self.BOARD_WIDTH = self.config['board_width']
            self.PANEL_WIDTH = self.config['panel_width']
            self.MENU_HEIGHT = self.config['menu_height']
            self.BORDER_THICKNESS = self.config['border_thickness']
            self.SQUARE_SIZE = self.config['square_size']
            self.BUTTON_SPACING_FROM_BOTTOM = self.config['button_spacing_from_bottom']
            self.PLAYER_IMAGE_SIZE = self.config['player_image_size']
            self.ANIMATION_FRAMES = self.config.get('animation_frames', 20)
            self.BLACK = (0, 0, 0)
            self.LIGHT_GRAY = (200, 200, 200)
            self.SKY_BLUE = (135, 206, 235)
            self.LIGHT_GREEN = (144, 238, 144)
            self.BLUE = (0, 0, 255)
            self.WHITE = (255, 255, 255)
            self.GRAY = (128, 128, 128)

            # تنظیم رنگ‌ها از config
            for key, value in self.config.items():
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

            # تعریف دکمه‌ها
            self.new_game_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 100,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[settings.language]["start_game"], self.SKY_BLUE
            )
            self.reset_scores_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 50,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[self.settings.language]["reset_scores"], self.LIGHT_GREEN
            )
            self.pause_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 150,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[settings.language]["pause"], self.LIGHT_GRAY
            )

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
            self._move_start_time = None
            self.hint_enabled_p1 = self.config['hint_enabled_p1_default']
            self.hint_enabled_p2 = self.config['hint_enabled_p2_default']
            self.hint_buttons = []
            self.hint_blink_timer = 0
            self.hint_blink_state = True
            self.undo_buttons = []

            log_to_json(
                "GameInterface initialized",
                level="INFO",
                extra_data={"settings": vars(self.settings), "window_size": [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]}
            )
        except Exception as e:
            log_to_json(
                f"Error initializing GameInterface: {str(e)}",
                level="ERROR",
                extra_data={"settings": vars(self.settings) if self.settings else None}
            )
            raise

    def apply_pending_settings(self):
        try:
            if self.pending_settings:
                for key, value in self.pending_settings.items():
                    if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                        value = hex_to_rgb(value)
                    setattr(self.settings, key, value)
                self.pending_settings = None
                self.load_player_images()
                self.load_piece_images()
                self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                self.reset_scores_button.text = LANGUAGES[self.settings.language]["reset_scores"]
                self.pause_button.text = LANGUAGES[self.settings.language]["pause"]
                self.game = Game(self.settings, self)
                self.screen = self.game.screen
                self.current_hand = 0
                self.auto_start_triggered = False
                log_to_json(
                    "Applied pending settings",
                    level="INFO",
                    extra_data={"settings": vars(self.settings)}
                )
        except Exception as e:
            log_to_json(
                f"Error applying pending settings: {str(e)}",
                level="ERROR",
                extra_data={"pending_settings": self.pending_settings}
            )

    def load_player_images(self):
        player_1_image_path = None  # مقداردهی اولیه
        player_2_image_path = None  # مقداردهی اولیه
        try:
            player_1_image_path = None  # مقداردهی اولیه
            player_2_image_path = None  # مقداردهی اولیه
            player_1_image_path = (
                self.settings.player_1_image
                if self.settings.game_mode in ["human_vs_human", "human_vs_ai"]
                else self.settings.al1_image
            )
            player_2_image_path = (
                self.settings.player_2_image
                if self.settings.game_mode == "human_vs_human"
                else self.settings.al2_image
            )
            # ادامه کد همانند بالا...
        except Exception as e:
            log_to_json(
                f"Error in load_player_images: {str(e)}",
                level="ERROR",
                extra_data={"player_1_image": player_1_image_path, "player_2_image": player_2_image_path}
            )

    def load_piece_images(self):
        try:
            piece_size = self.SQUARE_SIZE - 20
            for path, attr in [
                (self.settings.player_1_piece_image, 'player_1_piece_surface'),
                (self.settings.player_1_king_image, 'player_1_king_surface'),
                (self.settings.player_2_piece_image, 'player_2_piece_surface'),
                (self.settings.player_2_king_image, 'player_2_king_surface')
            ]:
                surface = None
                path_obj = None  # مقداردهی اولیه
                if path:
                    path_obj = project_root / path
                    if path_obj.exists() and path_obj.stat().st_size > 0:
                        try:
                            image = pygame.image.load(str(path_obj))
                            if image.get_size() != (0, 0):
                                image = pygame.transform.scale(image, (piece_size, piece_size))
                                mask = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                                pygame.draw.circle(
                                    mask, (255, 255, 255, 255),
                                    (piece_size // 2, piece_size // 2),
                                    piece_size // 2
                                )
                                image.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
                                surface = image
                            else:
                                log_to_json(
                                    f"Invalid image file {path}: Empty or corrupted",
                                    level="ERROR",
                                    extra_data={"path": str(path_obj)}
                                )
                                messagebox.showerror(
                                    LANGUAGES[self.settings.language]["error"],
                                    f"Invalid image file {path}: Empty or corrupted",
                                    parent=self.root
                                )
                        except pygame.error as e:
                            log_to_json(
                                f"Error loading image {path}: {str(e)}",
                                level="ERROR",
                                extra_data={"path": str(path_obj)}
                            )
                            messagebox.showerror(
                                LANGUAGES[self.settings.language]["error"],
                                f"Error loading image {path}: {e}",
                                parent=self.root
                            )
                setattr(self, attr, surface)
                log_to_json(
                    f"Loaded piece image for {attr}: {'success' if surface else 'default'}",
                    level="INFO",
                    extra_data={"path": str(path_obj) if path_obj else None}
                )
        except Exception as e:
            log_to_json(
                f"Error in load_piece_images: {str(e)}",
                level="ERROR",
                extra_data={"paths": [self.settings.player_1_piece_image, self.settings.player_1_king_image,
                                      self.settings.player_2_piece_image, self.settings.player_2_king_image]}
            )

    def draw_piece(self, screen, piece_value, row, col, is_removal=False, is_kinged=False):
        try:
            if piece_value == 0:
                return
            draw_x = col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            draw_y = row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            radius = self.SQUARE_SIZE // 2 - 10
            is_player_2 = piece_value < 0
            is_king = abs(piece_value) == 2 or is_kinged
            base_color = self.settings.player_2_color if is_player_2 else self.settings.player_1_color

            if is_removal:
                removal_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(removal_surface, (50, 50, 50, 100), (radius, radius), radius)
                screen.blit(removal_surface, (draw_x - radius, draw_y - radius))
                log_to_json(
                    f"Drew removal piece at ({row}, {col})",
                    level="DEBUG",
                    extra_data={"position": [row, col], "is_kinged": is_kinged}
                )
                return

            shadow_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surface, (50, 50, 50, 100), (radius + 5, radius + 5), radius)
            screen.blit(shadow_surface, (draw_x - radius - 5, draw_y - radius - 5))

            piece_surface = (
                self.player_2_king_surface if is_king else self.player_2_piece_surface
            ) if is_player_2 else (
                self.player_1_king_surface if is_king else self.player_1_piece_surface
            )

            if piece_surface is not None:
                screen.blit(piece_surface, (draw_x - radius, draw_y - radius))
            else:
                gradient_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                for r in range(radius, 0, -1):
                    t = r / radius
                    color = tuple(int(c * t + 255 * (1 - t)) for c in base_color)
                    if self.settings.piece_style == "circle":
                        pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                    elif self.settings.piece_style == "outlined_circle":
                        pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                    elif self.settings.piece_style == "square":
                        pygame.draw.rect(gradient_surface, color, (radius - r, radius - r, r * 2, r * 2))
                    elif self.settings.piece_style == "diamond":
                        points = [(radius, radius - r), (radius + r, radius), (radius, radius + r), (radius - r, radius)]
                        pygame.draw.polygon(gradient_surface, color, points)
                    elif self.settings.piece_style == "star":
                        points = [
                            (radius, radius - r),
                            (radius + r * 0.3, radius - r * 0.3),
                            (radius + r, radius),
                            (radius + r * 0.3, radius + r * 0.3),
                            (radius, radius + r),
                            (radius - r * 0.3, radius + r * 0.3),
                            (radius - r, radius),
                            (radius - r * 0.3, radius - r * 0.3)
                        ]
                        pygame.draw.polygon(gradient_surface, color, points)
                if self.settings.piece_style in ["outlined_circle", "diamond", "star"]:
                    if self.settings.piece_style == "outlined_circle":
                        pygame.draw.circle(gradient_surface, self.BLACK, (radius, radius), radius, 2)
                    elif self.settings.piece_style == "diamond":
                        points = [(radius, radius - radius), (radius + radius, radius), (radius, radius + radius),
                                  (radius - radius, radius)]
                        pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                    elif self.settings.piece_style == "star":
                        points = [
                            (radius, radius - radius),
                            (radius + radius * 0.3, radius - radius * 0.3),
                            (radius + radius, radius),
                            (radius + radius * 0.3, radius + radius * 0.3),
                            (radius, radius + radius),
                            (radius - radius * 0.3, radius + radius * 0.3),
                            (radius - radius, radius),
                            (radius - radius * 0.3, radius - radius * 0.3)
                        ]
                        pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                shine_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                for r in range(radius // 2, 0, -1):
                    alpha = int(100 * (r / (radius / 2)))
                    pygame.draw.circle(shine_surface, (255, 255, 255, alpha), (radius - radius // 4, radius - radius // 4),
                                       r)
                gradient_surface.blit(shine_surface, (0, 0))
                screen.blit(gradient_surface, (draw_x - radius, draw_y - radius))
                if is_king:
                    crown_radius = radius // 2
                    pygame.draw.circle(screen, self.GRAY, (draw_x, draw_y), crown_radius)
                    pygame.draw.circle(screen, self.BLACK, (draw_x, draw_y), crown_radius, 1)

            log_to_json(
                f"Drew piece at ({row}, {col})",
                level="DEBUG",
                extra_data={"piece_value": piece_value, "is_kinged": is_kinged, "is_player_2": is_player_2}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing piece at ({row}, {col}): {str(e)}",
                level="ERROR",
                extra_data={"piece_value": piece_value, "row": row, "col": col}
            )

    def animate_move(self, piece_value, start_row, start_col, end_row, end_col, is_removal=False, is_kinged=False):
        try:
            log_to_json(
                f"Animating move: piece={piece_value}, from ({start_row}, {start_col}) to ({end_row}, {end_col})",
                level="INFO",
                extra_data={"is_removal": is_removal, "is_kinged": is_kinged}
            )
            if piece_value == 0 or piece_value is None:
                log_to_json(
                    "Invalid piece_value! Animation aborted",
                    level="WARNING",
                    extra_data={"piece_value": piece_value}
                )
                return

            start_x = start_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            start_y = start_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            end_x = end_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            end_y = end_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            radius = self.SQUARE_SIZE // 2 - 10

            is_player_2 = piece_value < 0
            is_king = abs(piece_value) == 2 or is_kinged
            base_color = self.settings.player_2_color if is_player_2 else self.settings.player_1_color

            piece_surface = (
                self.player_2_king_surface if is_king else self.player_2_piece_surface
            ) if is_player_2 else (
                self.player_1_king_surface if is_king else self.player_1_piece_surface
            )

            if is_removal:
                for frame in range(self.ANIMATION_FRAMES + 1):
                    t = frame / self.ANIMATION_FRAMES
                    alpha = int(255 * (1 - t))
                    self.draw_game()
                    removal_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(removal_surface, (*base_color, alpha), (radius, radius), radius)
                    self.screen.blit(removal_surface, (start_x - radius, start_y - radius))
                    self.game.draw_valid_moves()
                    pygame.display.update()
                    pygame.time.wait(20)
                    log_to_json(
                        f"Animating removal frame {frame}/{self.ANIMATION_FRAMES}",
                        level="DEBUG",
                        extra_data={"alpha": alpha}
                    )
            else:
                for frame in range(self.ANIMATION_FRAMES + 1):
                    t = frame / self.ANIMATION_FRAMES
                    current_x = start_x + (end_x - start_x) * (1 - np.cos(t * np.pi)) / 2
                    current_y = start_y + (end_y - start_y) * (1 - np.cos(t * np.pi)) / 2
                    self.draw_game()
                    shadow_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
                    pygame.draw.circle(shadow_surface, (50, 50, 50, 100), (radius + 5, radius + 5), radius)
                    self.screen.blit(shadow_surface, (current_x - radius - 5, current_y - radius - 5))
                    if piece_surface is not None:
                        self.screen.blit(piece_surface, (current_x - radius, current_y - radius))
                    else:
                        gradient_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                        for r in range(radius, 0, -1):
                            t_r = r / radius
                            color = tuple(int(c * t_r + 255 * (1 - t_r)) for c in base_color)
                            if self.settings.piece_style == "circle":
                                pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                            elif self.settings.piece_style == "outlined_circle":
                                pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                            elif self.settings.piece_style == "square":
                                pygame.draw.rect(gradient_surface, color, (radius - r, radius - r, r * 2, r * 2))
                            elif self.settings.piece_style == "diamond":
                                points = [(radius, radius - r), (radius + r, radius), (radius, radius + r),
                                          (radius - r, radius)]
                                pygame.draw.polygon(gradient_surface, color, points)
                            elif self.settings.piece_style == "star":
                                points = [
                                    (radius, radius - r),
                                    (radius + r * 0.3, radius - r * 0.3),
                                    (radius + r, radius),
                                    (radius + r * 0.3, radius + r * 0.3),
                                    (radius, radius + r),
                                    (radius - r * 0.3, radius + r * 0.3),
                                    (radius - r, radius),
                                    (radius - r * 0.3, radius - r * 0.3)
                                ]
                                pygame.draw.polygon(gradient_surface, color, points)
                        if self.settings.piece_style in ["outlined_circle", "diamond", "star"]:
                            if self.settings.piece_style == "outlined_circle":
                                pygame.draw.circle(gradient_surface, self.BLACK, (radius, radius), radius, 2)
                            elif self.settings.piece_style == "diamond":
                                points = [(radius, radius - radius), (radius + radius, radius), (radius, radius + radius),
                                          (radius - radius, radius)]
                                pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                            elif self.settings.piece_style == "star":
                                points = [
                                    (radius, radius - radius),
                                    (radius + radius * 0.3, radius - radius * 0.3),
                                    (radius + radius, radius),
                                    (radius + radius * 0.3, radius + radius * 0.3),
                                    (radius, radius + radius),
                                    (radius - radius * 0.3, radius + radius * 0.3),
                                    (radius - radius, radius),
                                    (radius - radius * 0.3, radius - radius * 0.3)
                                ]
                                pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                        shine_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                        for r in range(radius // 2, 0, -1):
                            alpha = int(100 * (r / (radius / 2)))
                            pygame.draw.circle(shine_surface, (255, 255, 255, alpha),
                                               (radius - radius // 4, radius - radius // 4), r)
                        gradient_surface.blit(shine_surface, (0, 0))
                        self.screen.blit(gradient_surface, (current_x - radius, current_y - radius))
                        if is_kinged:
                            crown_radius = radius // 2
                            pygame.draw.circle(self.screen, self.GRAY, (current_x, current_y), crown_radius)
                            pygame.draw.circle(self.screen, self.BLACK, (current_x, current_y), crown_radius, 1)
                    self.game.draw_valid_moves()
                    pygame.display.update()
                    pygame.time.wait(20)
                    log_to_json(
                        f"Animating move frame {frame}/{self.ANIMATION_FRAMES}",
                        level="DEBUG",
                        extra_data={"current_position": [current_x, current_y]}
                    )
        except Exception as e:
            log_to_json(
                f"Error animating move: {str(e)}",
                level="ERROR",
                extra_data={
                    "piece_value": piece_value,
                    "start": [start_row, start_col],
                    "end": [end_row, end_col],
                    "is_removal": is_removal,
                    "is_kinged": is_kinged
                }
            )

    def draw_default_image(self, name, x, y):
        try:
            surface = pygame.Surface((self.PLAYER_IMAGE_SIZE, self.PLAYER_IMAGE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.LIGHT_GRAY,
                               (self.PLAYER_IMAGE_SIZE // 2, self.PLAYER_IMAGE_SIZE // 2),
                               self.PLAYER_IMAGE_SIZE // 2)
            text = self.small_font.render(name[:2], True, self.BLACK)
            text_rect = text.get_rect(center=(self.PLAYER_IMAGE_SIZE // 2, self.PLAYER_IMAGE_SIZE // 2))
            surface.blit(text, text_rect)
            self.screen.blit(surface, (x, y))
            log_to_json(
                f"Drew default image for {name} at ({x}, {y})",
                level="DEBUG",
                extra_data={"name": name, "position": [x, y]}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing default image for {name}: {str(e)}",
                level="ERROR",
                extra_data={"name": name, "position": [x, y]}
            )

    def draw_game(self):
        try:
            self.screen.fill(self.settings.board_color_1)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, 0, self.WINDOW_WIDTH, self.MENU_HEIGHT))

            # ترسیم منو
            for text_key, x in [
                ("settings", 10),
                ("ai_progress", 120),
                ("help", 250),
                ("about_me", 350)
            ]:
                text_surface = self.small_font.render(LANGUAGES[self.settings.language][text_key], True, self.BLACK)
                self.screen.blit(text_surface, (x, 5))
                #log_to_json(
                    #f"Drew menu text: {text_key}",
                    #level="DEBUG",
                    #extra_data={"text": text_key, "position": [x, 5]}
                #)

            pygame.draw.rect(self.screen, self.BLUE,
                             (0, self.MENU_HEIGHT, self.BOARD_WIDTH + self.BORDER_THICKNESS * 2,
                              self.BOARD_WIDTH + self.BORDER_THICKNESS * 3), self.BORDER_THICKNESS)
            self.game.board.draw(self.screen, self.settings.board_color_1, self.settings.board_color_2)

            # ترسیم مهره‌ها
            for row in range(8):
                for col in range(8):
                    piece_value = self.game.board.board[row, col]
                    if piece_value != 0:
                        self.draw_piece(self.screen, piece_value, row, col)

            # بررسی موقعیت ماوس و ترسیم حرکات معتبر
            mouse_pos = pygame.mouse.get_pos()
            hovered_moves = self.game.check_mouse_hover(mouse_pos)
            self.game.draw_valid_moves(hovered_moves)

            self.draw_side_panel()

            # ترسیم پاورقی
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, self.WINDOW_HEIGHT - 30, self.WINDOW_WIDTH, 30))
            footer_text = self.small_font.render(LANGUAGES[self.settings.language]["footer"], True, self.BLACK)
            footer_rect = footer_text.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT - 15))
            self.screen.blit(footer_text, footer_rect)
            log_to_json(
                "Drew game interface",
                level="DEBUG",
                extra_data={"board_state": self.game.board.board.tolist()}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing game interface: {str(e)}",
                level="ERROR",
                extra_data={"board_state": self.game.board.board.tolist() if hasattr(self.game, 'board') else None}
            )

    def draw_side_panel(self):
        player_1_name = None  # مقداردهی اولیه
        player_2_name = None  # مقداردهی اولیه
        try:
            scale = 0.85
            panel_rect = pygame.Rect(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2,
                self.MENU_HEIGHT,
                self.PANEL_WIDTH,
                self.WINDOW_HEIGHT - self.MENU_HEIGHT - 30
            )
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, panel_rect)
            y = self.MENU_HEIGHT + 20

            # عنوان نتیجه بازی
            result_title = self.font.render(LANGUAGES[self.settings.language]["game_result"], True, self.BLACK)
            result_rect = result_title.get_rect(
                center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y)
            )
            self.screen.blit(result_title, result_rect)
            y += int(40 * scale)

            # نمایش نتیجه بازی
            if self.game.game_over:
                if self.game.winner is None:
                    result_text = "بازی مساوی شد" if self.settings.language == "fa" else "Game is a Draw"
                elif not self.game.winner:
                    result_text = LANGUAGES[self.settings.language]["player_wins"]
                else:
                    result_text = LANGUAGES[self.settings.language]["player_wins"]
                result_display = self.small_font.render(result_text, True, self.BLACK)
                result_display_rect = result_display.get_rect(
                    center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y)
                )
                self.screen.blit(result_display, result_display_rect)
                log_to_json(
                    f"Drew game result: {result_text}",
                    level="INFO",
                    extra_data={"winner": self.game.winner}
                )
            y += int(70 * scale)

            # نام بازیکنان
            player_1_name = (
                self.settings.player_1_name
                if self.settings.game_mode in ["human_vs_human", "human_vs_ai"]
                else self.settings.al1_name
            )
            player_2_name = (
                self.settings.player_2_name
                if self.settings.game_mode == "human_vs_human"
                else self.settings.al2_name
            )

            # تصاویر بازیکنان
            scaled_image_size = int(self.PLAYER_IMAGE_SIZE * scale)
            if self.player_1_image_surface:
                scaled_image = pygame.transform.scale(self.player_1_image_surface, (scaled_image_size, scaled_image_size))
                self.screen.blit(
                    scaled_image,
                    (self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y)
                )
            else:
                self.draw_default_image(player_1_name, self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y)

            # متن "vs"
            vs_font = pygame.font.SysFont('Arial', int(24 * scale), bold=True)
            vs_text = vs_font.render("vs", True, self.BLUE)
            vs_text.set_alpha(200)
            vs_rect = vs_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2,
                    y + scaled_image_size // 2
                )
            )
            pygame.draw.rect(
                self.screen, self.LIGHT_GRAY,
                (vs_rect.x - int(5 * scale), vs_rect.y - int(5 * scale), vs_rect.width + int(10 * scale), vs_rect.height + int(10 * scale)),
                border_radius=int(5 * scale)
            )
            self.screen.blit(vs_text, vs_rect)

            # تصویر بازیکن دوم
            if self.player_2_image_surface:
                scaled_image = pygame.transform.scale(self.player_2_image_surface, (scaled_image_size, scaled_image_size))
                self.screen.blit(
                    scaled_image,
                    (
                        self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size,
                        y
                    )
                )
            else:
                self.draw_default_image(
                    player_2_name,
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size,
                    y
                )

            y += int(10 * scale)

            # نام‌های بازیکنان
            player_1_name_text = self.small_font.render(player_1_name, True, self.BLACK)
            player_1_name_rect = player_1_name_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20 + scaled_image_size // 2,
                    y + scaled_image_size + int(10 * scale)
                )
            )
            self.screen.blit(player_1_name_text, player_1_name_rect)

            player_2_name_text = self.small_font.render(player_2_name, True, self.BLACK)
            player_2_name_rect = player_2_name_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size // 2,
                    y + scaled_image_size + int(10 * scale)
                )
            )
            self.screen.blit(player_2_name_text, player_2_name_rect)

            y += int((scaled_image_size + 30) * scale)

            # جدول امتیازات
            table_rect = pygame.Rect(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y, int((self.PANEL_WIDTH - 40) * scale), int(80 * scale)
            )
            shadow_rect = pygame.Rect(table_rect.x + int(5 * scale), table_rect.y + int(5 * scale), table_rect.width, table_rect.height)
            pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=int(10 * scale))

            gradient_surface = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
            for i in range(table_rect.height):
                color = (255 - i * 2, 255 - i * 2, 255 - i * 2)
                pygame.draw.line(gradient_surface, color, (0, i), (table_rect.width, i))
            mask = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, table_rect.width, table_rect.height), border_radius=int(10 * scale))
            gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
            self.screen.blit(gradient_surface, (table_rect.x, table_rect.y))

            pygame.draw.rect(self.screen, self.BLACK, table_rect, 2, border_radius=int(10 * scale))
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x + table_rect.width // 3, table_rect.y),
                (table_rect.x + table_rect.width // 3, table_rect.y + table_rect.height), 2
            )
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x + 2 * table_rect.width // 3, table_rect.y),
                (table_rect.x + 2 * table_rect.width // 3, table_rect.y + table_rect.height), 2
            )
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x, table_rect.y + table_rect.height // 2),
                (table_rect.x + table_rect.width, table_rect.y + table_rect.height // 2), 2
            )

            wins_1 = self.small_font.render(str(self.game.player_1_wins), True, self.BLACK)
            wins_1_rect = wins_1.get_rect(
                center=(table_rect.x + table_rect.width // 6, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_1, wins_1_rect)

            wins_text = self.small_font.render(LANGUAGES[self.settings.language]["wins"], True, self.BLACK)
            wins_text_rect = wins_text.get_rect(
                center=(table_rect.x + table_rect.width // 2, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_text, wins_text_rect)

            wins_2 = self.small_font.render(str(self.game.player_2_wins), True, self.BLACK)
            wins_2_rect = wins_2.get_rect(
                center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_2, wins_2_rect)

            pieces_1 = self.small_font.render(str(self.game.board.player_1_left), True, self.BLACK)
            pieces_1_rect = pieces_1.get_rect(
                center=(table_rect.x + table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_1, pieces_1_rect)

            pieces_text = self.small_font.render(LANGUAGES[self.settings.language]["pieces"], True, self.BLACK)
            pieces_text_rect = pieces_text.get_rect(
                center=(table_rect.x + table_rect.width // 2, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_text, pieces_text_rect)

            pieces_2 = self.small_font.render(str(self.game.board.player_2_left), True, self.BLACK)
            pieces_2_rect = pieces_2.get_rect(
                center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_2, pieces_2_rect)

            y += int(110 * scale)

            # عنوان زمان
            time_title = self.small_font.render(LANGUAGES[self.settings.language]["time"], True, self.BLACK)
            time_title_rect = time_title.get_rect(
                center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y))
            self.screen.blit(time_title, time_title_rect)
            y += int(20 * scale)

            # تایمر
            timer_rect = pygame.Rect(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y, int((self.PANEL_WIDTH - 40) * scale), int(40 * scale))
            shadow_rect = pygame.Rect(timer_rect.x + int(5 * scale), timer_rect.y + int(5 * scale), timer_rect.width, timer_rect.height)
            pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=int(10 * scale))
            gradient_surface = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
            for i in range(timer_rect.height):
                color = (135 - i * 2, 206 - i * 2, 235 - i * 2)
                pygame.draw.line(gradient_surface, color, (0, i), (timer_rect.width, i))
            mask = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, timer_rect.width, timer_rect.height), border_radius=int(10 * scale))
            gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
            self.screen.blit(gradient_surface, (timer_rect.x, timer_rect.y))

            pygame.draw.rect(self.screen, self.BLACK, timer_rect, 2, border_radius=int(10 * scale))
            pygame.draw.line(self.screen, self.BLACK, (timer_rect.x + timer_rect.width // 2, timer_rect.y),
                             (timer_rect.x + timer_rect.width // 2, timer_rect.y + timer_rect.height), 2)

            bold_font = pygame.font.SysFont('Arial', int(18 * scale), bold=True)
            player_1_time = self.game.timer.get_current_time(False)
            player_2_time = self.game.timer.get_current_time(True)
            timer_1 = bold_font.render(f"{int(player_1_time)} s", True, self.BLACK if not self.game.turn else self.GRAY)
            timer_1_rect = timer_1.get_rect(
                center=(timer_rect.x + timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
            self.screen.blit(timer_1, timer_1_rect)

            timer_2 = bold_font.render(f"{int(player_2_time)} s", True, self.BLACK if self.game.turn else self.GRAY)
            timer_2_rect = timer_2.get_rect(
                center=(timer_rect.x + 3 * timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
            self.screen.blit(timer_2, timer_2_rect)

            y += int(60 * scale)

            # دکمه‌های Undo و Redo
            undo_button_x = (
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 +
                    (self.PANEL_WIDTH - int(2 * self.config['undo_button_width'] * scale) - int(self.config['undo_redo_button_spacing'] * scale)) // 2
            )
            undo_button_y = y + int(self.config['undo_redo_y_offset'] * scale)

            undo_button = Button(
                undo_button_x, undo_button_y, int(self.config['undo_button_width'] * scale), int(self.config['undo_button_height'] * scale),
                LANGUAGES[self.settings.language]["undo"], self.SKY_BLUE
            )
            undo_button.enabled = len(self.game.history.move_history) >= 2

            redo_button = Button(
                undo_button_x + int(self.config['undo_button_width'] * scale) + int(self.config['undo_redo_button_spacing'] * scale), undo_button_y,
                int(self.config['redo_button_width'] * scale), int(self.config['redo_button_height'] * scale),
                LANGUAGES[self.settings.language]["redo"], self.SKY_BLUE
            )
            redo_button.enabled = len(self.game.history.redo_stack) > 0

            self.undo_buttons = [undo_button, redo_button]
            for button in self.undo_buttons:
                button.draw(self.screen)

            y += int((self.config['undo_button_height'] + self.config['undo_redo_y_offset'] + 10) * scale)

            # دکمه‌های Hint
            hint_button_x = (
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 +
                    (self.PANEL_WIDTH - int(2 * self.config['hint_button_width'] * scale) - int(self.config['hint_button_spacing'] * scale)) // 2
            )
            hint_button_y = y + int(self.config['hint_button_y_offset'] * scale)

            p1_hint_text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p1 else "hint_off"]
            p1_hint_button = Button(
                hint_button_x, hint_button_y, int(self.config['hint_button_width'] * scale), int(self.config['hint_button_height'] * scale),
                p1_hint_text, self.SKY_BLUE
            )

            p2_hint_text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p2 else "hint_off"]
            p2_hint_button = Button(
                hint_button_x + int(self.config['hint_button_width'] * scale) + int(self.config['hint_button_spacing'] * scale), hint_button_y,
                int(self.config['hint_button_width'] * scale), int(self.config['hint_button_height'] * scale),
                p2_hint_text, self.SKY_BLUE
            )

            self.hint_buttons = [p1_hint_button, p2_hint_button]

            if self.settings.game_mode == "human_vs_ai" and not self.game.turn:
                p1_hint_button.enabled = True
                p2_hint_button.enabled = False
            elif self.settings.game_mode == "human_vs_ai" and self.game.turn:
                p1_hint_button.enabled = False
                p2_hint_button.enabled = True
            elif self.settings.game_mode == "human_vs_human":
                p1_hint_button.enabled = True
                p2_hint_button.enabled = True
            else:
                p1_hint_button.enabled = False
                p2_hint_button.enabled = False

            for button in self.hint_buttons:
                button.draw(self.screen)

            y += int((self.config['hint_button_height'] + self.config['hint_button_y_offset'] + 10) * scale)

            # دکمه‌های بازی جدید، بازنشانی امتیازات و توقف
            self.pause_button.draw(self.screen)
            self.new_game_button.draw(self.screen)
            self.reset_scores_button.draw(self.screen)

            log_to_json(
                "Drew side panel",
                level="DEBUG",
                extra_data={"player_1_name": player_1_name, "player_2_name": player_2_name}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing side panel: {str(e)}",
                level="ERROR",
                extra_data={"player_1_name": player_1_name, "player_2_name": player_2_name}
            )

    def _handle_events(self, settings_window, ai_progress_window, help_window, about_window):
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    log_to_json("Quit event received", level="INFO")
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    log_to_json(
                        f"Mouse click at {pos}",
                        level="DEBUG",
                        extra_data={"pos": list(pos)}
                    )
                    if pos[1] < self.MENU_HEIGHT:
                        if not (settings_window.is_open or ai_progress_window.is_open or
                                help_window.is_open or about_window.is_open):
                            if pos[0] < 120:
                                logger.info("Opening settings_window")
                                settings_window.create_widgets()
                                settings_window.is_open = True
                            elif 120 <= pos[0] < 250:
                                logger.info("Opening ai_progress_window")
                                ai_progress_window.create_widgets()
                                ai_progress_window.is_open = True
                            elif 250 <= pos[0] < 350:
                                logger.info("Opening help_window")
                                help_window.create_widgets()
                                help_window.is_open = True
                            elif 350 <= pos[0] < 450:
                                logger.info("Opening about_window")
                                about_window.create_widgets()
                                about_window.is_open = True
                    elif self.new_game_button.is_clicked(pos):
                        logger.info("New game button clicked")
                        if not self.game.game_started:
                            logger.info("Starting new game")
                            self.game.start_game()
                            self.new_game_button.text = LANGUAGES[self.settings.language]["new_game"]
                            self.last_update = pygame.time.get_ticks()
                            self.current_hand = 0
                            self.auto_start_triggered = False
                            sound_path = project_root / "assets" / "start.wav"
                            if self.settings.sound_enabled and sound_path.exists() and sound_path.stat().st_size > 0:
                                try:
                                    pygame.mixer.Sound(str(sound_path)).play()
                                    log_to_json("Played start sound", level="INFO", extra_data={"sound_path": str(sound_path)})
                                except pygame.error as e:
                                    log_to_json(
                                        f"Error playing start.wav: {str(e)}",
                                        level="ERROR",
                                        extra_data={"sound_path": str(sound_path)}
                                    )
                        elif self.game.game_started and not self.game.game_over:
                            confirmed = messagebox.askyesno(
                                LANGUAGES[self.settings.language]["warning"],
                                LANGUAGES[self.settings.language]["new_game_warning"],
                                parent=self.root
                            )
                            if confirmed:
                                logger.info("Restarting game")
                                self.game.game_over = True
                                self.game.score_updated = False
                                self.game.init_game()
                                self.game.game_started = False
                                self.game.game_over = False
                                self.game.winner = None
                                self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                                self.last_update = pygame.time.get_ticks()
                                self.current_hand = 0
                                self.auto_start_triggered = False
                                pygame.event.clear()
                        elif self.game.game_over:
                            logger.info("Starting new game after game over")
                            self.game.game_over = True
                            self.game.score_updated = False
                            self.game.init_game()
                            self.game.game_started = False
                            self.game.game_over = False
                            self.game.winner = None
                            self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                            self.last_update = pygame.time.get_ticks()
                            self.current_hand = 0
                            self.auto_start_triggered = False
                            pygame.event.clear()
                    elif self.reset_scores_button.is_clicked(pos):
                        logger.info("Reset scores button clicked")
                        if messagebox.askyesno(
                                LANGUAGES[self.settings.language]["warning"],
                                LANGUAGES[self.settings.language]["reset_scores_warning"],
                                parent=self.root
                        ):
                            self.game.player_1_wins = 0
                            self.game.player_2_wins = 0
                            save_stats({"player_1_wins": 0, "player_2_wins": 0})
                            log_to_json(
                                "Scores reset",
                                level="INFO",
                                extra_data={"player_1_wins": 0, "player_2_wins": 0}
                            )
                    elif self.pause_button.is_clicked(pos) and self.game.game_started and not self.game.game_over:
                        logger.info("Pause button clicked")
                        self.game.paused = not self.game.paused
                        if self.game.paused:
                            self.game.timer.pause()
                            self.pause_button.text = LANGUAGES[self.settings.language]["resume"]
                            log_to_json("Game paused", level="INFO")
                        else:
                            self.game.timer.unpause()
                            self.pause_button.text = LANGUAGES[self.settings.language]["pause"]
                            log_to_json("Game resumed", level="INFO")
                    elif self.game.game_started:
                        for button in self.undo_buttons:
                            if hasattr(button, 'enabled') and button.enabled and button.is_clicked(pos):
                                if button.text == LANGUAGES[self.settings.language]["undo"]:
                                    self.undo_move()
                                elif button.text == LANGUAGES[self.settings.language]["redo"]:
                                    self.redo_move()
                        for button in self.hint_buttons:
                            if hasattr(button, 'enabled') and button.enabled and button.is_clicked(pos):
                                if button == self.hint_buttons[0]:
                                    self.toggle_hint_p1()
                                elif button == self.hint_buttons[1]:
                                    self.toggle_hint_p2()
                        logger.debug("Handling board click")
                        self.game.handle_click(pos)
            return True
        except Exception as e:
            log_to_json(
                f"Error handling events: {str(e)}",
                level="ERROR",
                extra_data={"event_type": None}
            )
            return True

    def update(self):
        try:
            current_time = pygame.time.get_ticks()
            self.game.update_game()
            if self.game.game_started and not self.game.game_over:
                if self.settings.game_mode == "ai_vs_ai":
                    if self.player_1_move_ready and not self.game.turn:
                        logger.debug("Player 1 AI move ready")
                        if "ai_1" in self.game.ai_players:
                            self.player_1_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_1")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_1",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_1",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_1_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_1")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_1",
                                level="ERROR",
                                extra_data={"ai_id": "ai_1", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif self.player_2_move_ready and self.game.turn:
                        logger.debug("Player 2 AI move ready")
                        if "ai_2" in self.game.ai_players:
                            self.player_2_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_2")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_2",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_2",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_2_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_2")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_2",
                                level="ERROR",
                                extra_data={"ai_id": "ai_2", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif not self.player_1_move_ready and not self.game.turn:
                        if current_time - self._move_start_time >= self.settings.ai_pause_time:
                            logger.debug("Player 1 AI move ready after pause")
                            self.player_1_move_ready = True
                    elif not self.player_2_move_ready and self.game.turn:
                        if current_time - self._move_start_time >= self.settings.ai_pause_time:
                            logger.debug("Player 2 AI move ready after pause")
                            self.player_2_move_ready = True
                elif self.settings.game_mode == "human_vs_ai" and self.game.turn:
                    if self.player_2_move_ready:
                        logger.debug("Player 2 AI move ready (human_vs_ai)")
                        if "ai_2" in self.game.ai_players:
                            self.player_2_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_2")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_2 (human_vs_ai)",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_2",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_2_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_2 (human_vs_ai)")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_2 (human_vs_ai)",
                                level="ERROR",
                                extra_data={"ai_id": "ai_2", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif current_time - self._move_start_time >= self.settings.ai_pause_time:
                        logger.debug("Player 2 AI move ready after pause (human_vs_ai)")
                        self.player_2_move_ready = True
                if self.game.game_over and self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game":
                    if self.current_hand < self.settings.repeat_hands:
                        if self.pause_start_time is None:
                            self.pause_start_time = current_time
                        elif current_time - self.pause_start_time >= self.settings.pause_between_hands:
                            logger.info("Starting new hand")
                            self.game.reset_board()
                            self.game.start_game()
                            self.current_hand += 1
                            self.pause_start_time = None
                            self.auto_start_triggered = False
                    else:
                        logger.info("All hands completed")
                        self.auto_start_triggered = True
                        self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                        self.game.game_started = False
            self.draw_game()
            pygame.display.update()
            log_to_json(
                "Game updated",
                level="DEBUG",
                extra_data={"game_started": self.game.game_started, "game_over": self.game.game_over}
            )
        except Exception as e:
            log_to_json(
                f"Error updating game: {str(e)}",
                level="ERROR",
                extra_data={"game_started": self.game.game_started, "game_over": self.game.game_over}
            )

    def close_windows(self, settings_window, ai_progress_window, help_window, about_window):
        try:
            for window in [settings_window, ai_progress_window, help_window, about_window]:
                if window is not None and window.is_open:
                    window.close()
                    log_to_json(
                        f"Closed window: {window.__class__.__name__}",
                        level="INFO"
                    )
            self.root.destroy()
            log_to_json("Tkinter root destroyed", level="INFO")
        except tk.TclError as e:
            log_to_json(
                f"TclError closing windows: {str(e)}",
                level="ERROR"
            )

    def undo_move(self):
        try:
            if len(self.game.history.move_history) < 1:
                logger.info("No moves to undo")
                return
            self.game.history.undo(self.game)
            if self.game.multi_jump_active and self.game.selected:
                valid_moves = self.game.get_valid_moves(*self.game.selected)
                self.game.valid_moves = {(move[2], move[3]): skipped for move, skipped in valid_moves.items() if skipped}
            elif not self.game.multi_jump_active:
                self.game.valid_moves = {}
            self.game.draw_valid_moves()
            log_to_json(
                "Undo move performed",
                level="INFO",
                extra_data={"move_history_length": len(self.game.history.move_history)}
            )
        except Exception as e:
            log_to_json(
                f"Error undoing move: {str(e)}",
                level="ERROR",
                extra_data={"move_history_length": len(self.game.history.move_history)}
            )

    def redo_move(self):
        try:
            self.game.history.redo(self.game)
            self.game.draw_valid_moves()
            log_to_json(
                "Redo move performed",
                level="INFO",
                extra_data={"redo_stack_length": len(self.game.history.redo_stack)}
            )
        except Exception as e:
            log_to_json(
                f"Error redoing move: {str(e)}",
                level="ERROR",
                extra_data={"redo_stack_length": len(self.game.history.redo_stack)}
            )

    def toggle_hint_p1(self):
        try:
            self.hint_enabled_p1 = not self.hint_enabled_p1
            self.hint_buttons[0].text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p1 else "hint_off"]
            log_to_json(
                f"Toggled hint for player 1: {self.hint_enabled_p1}",
                level="INFO",
                extra_data={"hint_enabled_p1": self.hint_enabled_p1}
            )
        except Exception as e:
            log_to_json(
                f"Error toggling hint for player 1: {str(e)}",
                level="ERROR"
            )

    def toggle_hint_p2(self):
        try:
            self.hint_enabled_p2 = not self.hint_enabled_p2
            self.hint_buttons[1].text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p2 else "hint_off"]
            log_to_json(
                f"Toggled hint for player 2: {self.hint_enabled_p2}",
                level="INFO",
                extra_data={"hint_enabled_p2": self.hint_enabled_p2}
            )
        except Exception as e:
            log_to_json(
                f"Error toggling hint for player 2: {str(e)}",
                level="ERROR"
            )

    def draw_hint(self, start_row, start_col, end_row, end_col):
        try:
            if (start_row is None or start_col is None or end_row is None or end_col is None) or not (
                    self.hint_enabled_p1 and not self.game.turn or self.hint_enabled_p2 and self.game.turn):
                log_to_json(
                    "Skipping hint drawing: Invalid coordinates or hint disabled",
                    level="DEBUG",
                    extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
                )
                return
            current_time = pygame.time.get_ticks()
            if current_time - self.hint_blink_timer > self.config['hint_blink_interval']:
                self.hint_blink_state = not self.hint_blink_state
                self.hint_blink_timer = current_time
            if not self.hint_blink_state:
                return
            start_x = start_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            start_y = start_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            end_x = end_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            end_y = end_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            pygame.draw.circle(self.screen, self.config['hint_circle_color'], (start_x, start_y),
                               self.config['hint_circle_radius'])
            pygame.draw.circle(self.screen, self.config['hint_circle_color'], (end_x, end_y),
                               self.config['hint_circle_radius'])
            log_to_json(
                f"Drew hint from ({start_row}, {start_col}) to ({end_row}, {end_col})",
                level="DEBUG",
                extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing hint: {str(e)}",
                level="ERROR",
                extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
            )

    def run(self):
        settings_window = SettingsWindow(self, self.root)
        ai_progress_window = AIProgressWindow(self, self.root)
        help_window = HelpWindow(self, self.root)
        about_window = AboutWindow(self, self.root)

        clock = pygame.time.Clock()

        running = True
        while running:
            running = self._handle_events(settings_window, ai_progress_window, help_window,
                                          about_window)

            self.update()

            self.screen.fill(self.settings.board_color_1)
            self.draw_game()

            if self.settings.game_mode in ["human_vs_human", "human_vs_ai"] and (
                    self.hint_enabled_p1 or self.hint_enabled_p2):
                hint = self.game.get_hint()  # استفاده از get_hint از game.py
                if hint is not None:
                    start_row, start_col, end_row, end_col = hint
                    self.draw_hint(start_row, start_col, end_row, end_col)
                else:
                    self.draw_hint(None, None, None, None)

            pygame.display.flip()

            try:
                self.root.update()
            except tk.TclError:
                running = False

            clock.tick(self.config.get('fps', 60))

        self.close_windows(settings_window, ai_progress_window, help_window, about_window)
        pygame.quit()
        sys.exit()