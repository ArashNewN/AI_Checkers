import pygame
from .constants import LANGUAGES

class TimerManager:
    def __init__(self, use_timer, game_time):
        self.use_timer = use_timer
        self.game_time = game_time * 60
        self.player_1_time = self.game_time if use_timer else 0
        self.player_2_time = self.game_time if use_timer else 0
        self.game_started = False
        self.last_update = pygame.time.get_ticks()
        self.paused = False

    def start_game(self):
        self.game_started = True
        self.paused = False
        self.last_update = pygame.time.get_ticks()
        if self.use_timer:
            self.player_1_time = self.game_time
            self.player_2_time = self.game_time
        else:
            self.player_1_time = 0
            self.player_2_time = 0

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False
        self.last_update = pygame.time.get_ticks()

    def update(self, turn, game_started, game_over):
        if not self.game_started or game_over or self.paused:
            return
        if not game_started:
            return
        current_time = pygame.time.get_ticks()
        delta_time = (current_time - self.last_update) / 1000.0
        self.last_update = current_time
        if self.use_timer:
            if not turn:
                self.player_1_time = max(0, self.player_1_time - delta_time)
            else:
                self.player_2_time = max(0, self.player_2_time - delta_time)
        else:
            if not turn:
                self.player_1_time += delta_time
            else:
                self.player_2_time += delta_time

    def get_current_time(self, is_player_2):
        if self.use_timer:
            return self.player_2_time if is_player_2 else self.player_1_time
        else:
            return self.player_2_time if is_player_2 else self.player_1_time