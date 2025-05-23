#board.py
import os
import numpy as np
import pygame
from typing import Dict, Optional
from .utils import CheckersError
from .config import ConfigManager, log_to_json
from .constants import SQUARE_SIZE, BORDER_THICKNESS, MENU_HEIGHT

# نمونه جهانی ConfigManager
_config_manager = ConfigManager()

class Board:
    def __init__(self, settings):
        config = _config_manager.load_config()
        self.board_size = config.get("board_size", 8)  # مقدار پیش‌فرض 8
        if self.board_size % 2 != 0 or self.board_size < 4:
            log_to_json(
                f"Invalid board_size: {self.board_size}. Must be even and >= 4.",
                level="ERROR",
                extra_data={"board_size": self.board_size}
            )
            raise CheckersError(f"Invalid board_size: {self.board_size}")
        self.settings = settings
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        initial_rows = self.board_size // 2 - 1
        pieces_per_row = self.board_size // 2
        self.player_2_left = initial_rows * pieces_per_row
        self.player_1_left = initial_rows * pieces_per_row
        self.create_board()
        # اصلاح نوع دیکشنری برای پشتیبانی از Surface یا None
        self.piece_images: Dict[int, Optional[pygame.Surface]] = {1: None, 2: None, -1: None, -2: None}
        self.load_piece_images()

    def copy(self):
        """Returns a deep copy of the board."""
        new_board = Board(self.settings)
        new_board.board = self.board.copy()
        new_board.player_1_left = self.player_1_left
        new_board.player_2_left = self.player_2_left
        new_board.piece_images = self.piece_images.copy()  # کپی دیکشنری تصاویر
        return new_board

    def load_piece_images(self):
        """Loads piece images from paths specified in settings."""
        config = _config_manager.load_config()
        assets_dir = config.get("assets_dir", "assets")
        project_root = _config_manager.get_project_root()
        image_paths = {
            1: self.settings.player_1_piece_image or str(project_root / assets_dir / "white_piece.png"),
            2: self.settings.player_1_king_image or str(project_root / assets_dir / "white_king.png"),
            -1: self.settings.player_2_piece_image or str(project_root / assets_dir / "black_piece.png"),
            -2: self.settings.player_2_king_image or str(project_root / assets_dir / "black_king.png")
        }
        for piece_type, path in image_paths.items():
            if path and os.path.exists(path):
                try:
                    image = pygame.image.load(path)
                    self.piece_images[piece_type] = pygame.transform.scale(
                        image, (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
                    )
                except pygame.error as e:
                    log_to_json(
                        f"Failed to load image for piece {piece_type}: {str(e)}",
                        level="ERROR",
                        extra_data={"piece_type": piece_type, "path": path}
                    )
            #else:
                #log_to_json(
                    #f"Image path for piece {piece_type} does not exist: {path}",
                    #level="WARNING",
                    #extra_data={"piece_type": piece_type, "path": path}
                #)

    def create_board(self):
        """Initializes the board with pieces for both players."""
        initial_rows = self.board_size // 2 - 1
        for row in range(initial_rows):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1  # مهره‌های بازیکن 2
        for row in range(self.board_size - initial_rows, self.board_size):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1  # مهره‌های بازیکن 1
        log_to_json(
            f"Board created with size {self.board_size}x{self.board_size}",
            level="DEBUG",
            extra_data={"board": self.board.tolist()}
        )

    def draw(self, screen, board_color_1, board_color_2):
        """Draws the board and pieces on the screen."""
        config = _config_manager.load_config()
        assets_dir = config.get("assets_dir", "assets")
        project_root = _config_manager.get_project_root()
        light_texture_path = project_root / assets_dir / "wood_texture_light.png"
        dark_texture_path = project_root / assets_dir / "wood_texture_dark.png"
        light_texture = None
        dark_texture = None
        try:
            if light_texture_path.exists():
                light_texture = pygame.image.load(str(light_texture_path))
                light_texture = pygame.transform.scale(light_texture, (SQUARE_SIZE, SQUARE_SIZE))
            if dark_texture_path.exists():
                dark_texture = pygame.image.load(str(dark_texture_path))
                dark_texture = pygame.transform.scale(dark_texture, (SQUARE_SIZE, SQUARE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            log_to_json(
                f"Failed to load textures: {str(e)}",
                level="WARNING",
                extra_data={"light_texture": str(light_texture_path), "dark_texture": str(dark_texture_path)}
            )

        for row in range(self.board_size):
            for col in range(self.board_size):
                x = col * SQUARE_SIZE + BORDER_THICKNESS
                y = row * SQUARE_SIZE + MENU_HEIGHT + BORDER_THICKNESS
                if (row + col) % 2 == 0:
                    if light_texture:
                        screen.blit(light_texture, (x, y))
                    else:
                        pygame.draw.rect(screen, board_color_1, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                else:
                    if dark_texture:
                        screen.blit(dark_texture, (x, y))
                    else:
                        pygame.draw.rect(screen, board_color_2, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                piece = int(self.board[row, col])  # تبدیل به int برای رفع خطای نوع
                if piece != 0:
                    center_x = x + SQUARE_SIZE // 2
                    center_y = y + SQUARE_SIZE // 2
                    if self.piece_images.get(piece):
                        image_rect = self.piece_images[piece].get_rect(center=(center_x, center_y))
                        screen.blit(self.piece_images[piece], image_rect)
                    else:
                        color = self.settings.player_1_color if piece > 0 else self.settings.player_2_color
                        pygame.draw.circle(screen, color, (center_x, center_y), SQUARE_SIZE // 3)
                        if abs(piece) == 2:
                            pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), SQUARE_SIZE // 6)

    def remove(self, pieces):
        """Removes pieces from the board and updates piece counts."""
        for row, col in pieces:
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                log_to_json(
                    f"Invalid position for removal: ({row}, {col})",
                    level="ERROR",
                    extra_data={"row": row, "col": col, "board_size": self.board_size}
                )
                raise CheckersError(f"Invalid position: ({row}, {col})")
            if self.board[row, col] != 0:
                if self.board[row, col] < 0:
                    self.player_2_left -= 1
                else:
                    self.player_1_left -= 1
                self.board[row, col] = 0
        log_to_json(
            f"Removed pieces: {pieces}, player_1_left={self.player_1_left}, player_2_left={self.player_2_left}",
            level="DEBUG",
            extra_data={"pieces": pieces}
        )