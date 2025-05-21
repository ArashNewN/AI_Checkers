# board.py
import os
import numpy as np
import pygame
from .utils import CheckersError
from .config import load_config
from .constants import SQUARE_SIZE, BORDER_THICKNESS, MENU_HEIGHT


class Board:
    def __init__(self, settings):
        config = load_config()
        self.board_size = config["board_size"]
        self.settings = settings
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        initial_rows = self.board_size // 2 - 1
        pieces_per_row = self.board_size // 2
        self.player_2_left = initial_rows * pieces_per_row
        self.player_1_left = initial_rows * pieces_per_row
        self.create_board()
        self.piece_images = {1: None, 2: None, -1: None, -2: None}
        self.load_piece_images()

    def copy(self):
        """Returns a deep copy of the board."""
        new_board = Board(self.settings)
        new_board.board = self.board.copy()
        new_board.player_1_left = self.player_1_left
        new_board.player_2_left = self.player_2_left
        return new_board

    def load_piece_images(self):
        image_paths = {
            1: self.settings.player_1_piece_image,
            2: self.settings.player_1_king_image,
            -1: self.settings.player_2_piece_image,
            -2: self.settings.player_2_king_image
        }
        for piece_type, path in image_paths.items():
            if path and os.path.exists(path):
                try:
                    image = pygame.image.load(path)
                    self.piece_images[piece_type] = pygame.transform.scale(image, (SQUARE_SIZE // 2, SQUARE_SIZE // 2))
                except pygame.error as e:
                    from .checkers_core import log_to_json
                    log_to_json(f"Failed to load image for piece {piece_type}: {str(e)}", "ERROR")

    def create_board(self):
        initial_rows = self.board_size // 2 - 1
        for row in range(initial_rows):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    self.board[row, col] = -1
        for row in range(self.board_size - initial_rows, self.board_size):
            for col in range(self.board_size):
                if (row + col) % 2 == 1:
                    self.board[row, col] = 1

    def draw(self, screen, board_color_1, board_color_2):
        light_texture = None
        dark_texture = None
        try:
            light_texture = pygame.image.load("wood_texture_light.png")
            dark_texture = pygame.image.load("wood_texture_dark.png")
            light_texture = pygame.transform.scale(light_texture, (SQUARE_SIZE, SQUARE_SIZE))
            dark_texture = pygame.transform.scale(dark_texture, (SQUARE_SIZE, SQUARE_SIZE))
        except (pygame.error, FileNotFoundError):
            pass

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
                piece = self.board[row, col]
                if piece != 0:
                    center_x = x + SQUARE_SIZE // 2
                    center_y = y + SQUARE_SIZE // 2
                    if self.piece_images[piece]:
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
                raise CheckersError(f"Invalid position: ({row}, {col})")
            if self.board[row, col] != 0:
                if self.board[row, col] < 0:
                    self.player_2_left -= 1
                else:
                    self.player_1_left -= 1
                self.board[row, col] = 0