import pygame
from .constants import BOARD_SIZE, SQUARE_SIZE, BORDER_THICKNESS, MENU_HEIGHT

class Piece:
    def __init__(self, row, col, is_player_2, color):
        self.row = row
        self.col = col
        self.is_player_2 = is_player_2
        self.color = color
        self.king = False
        self.x = col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS
        self.y = row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS

    def make_king(self):
        self.king = True

    def move(self, row, col):
        self.row = row
        self.col = col
        self.x = col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS
        self.y = row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS

class Board:
    def __init__(self, settings):
        self.settings = settings
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.player_2_left = 12
        self.player_1_left = 12
        self.create_board()

    def create_board(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if col % 2 == ((row + 1) % 2):
                    if row < 3:
                        self.board[row][col] = Piece(row, col, True, self.settings.player_2_color)
                    elif row > 4:
                        self.board[row][col] = Piece(row, col, False, self.settings.player_1_color)

    def draw(self, screen, board_color_1, board_color_2):
        light_texture = None
        dark_texture = None
        try:
            # بارگذاری تکسچرهای چوب
            light_texture = pygame.image.load("wood_texture_light.png")
            dark_texture = pygame.image.load("wood_texture_dark.png")
            light_texture = pygame.transform.scale(light_texture, (SQUARE_SIZE, SQUARE_SIZE))
            dark_texture = pygame.transform.scale(dark_texture, (SQUARE_SIZE, SQUARE_SIZE))
        except (pygame.error, FileNotFoundError):
            # در صورت نبود تکسچر، از رنگ‌های تنظیمات استفاده می‌کنیم
            pass

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
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

    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = None
            if piece.is_player_2:
                self.player_2_left -= 1
            else:
                self.player_1_left -= 1