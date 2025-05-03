import json
import os
import random
import sys
from collections import deque
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tkinter import ttk, messagebox, filedialog, colorchooser
import tkinter as tk

SQUARE_SIZE = 80
BOARD_SIZE = 8
BORDER_THICKNESS = 7
MENU_HEIGHT = 30
WINDOW_WIDTH = BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 2 + 300
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
WINDOW_HEIGHT = BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 3 + 60
PANEL_WIDTH = 300
BUTTON_SPACING_FROM_BOTTOM = 40
ANIMATION_FRAMES = 1
PLAYER_IMAGE_SIZE = int(PANEL_WIDTH * 0.25)
GAME_VERSION = "1.0"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)
LIGHT_GRAY = (211, 211, 211)
LIGHT_GREEN = (144, 238, 144)

CONFIG_FILE = 'settings.json'
STATS_FILE = 'game_stats.json'

LANGUAGES = {
    "en": {
        "piece_images": "Piece Images",
        "player_1_piece": "Player 1 Piece",
        "player_1_king": "Player 1 King",
        "player_2_piece": "Player 2 Piece",
        "player_2_king": "Player 2 King",
        "piece_images": "Piece Images",
        "player_1_piece": "Player 1 Piece",
        "player_1_king": "Player 1 King",
        "player_2_piece": "Player 2 Piece",
        "player_2_king": "Player 2 King",
        "al_progress": "AL Progress",
        "az_progress": "AZ Progress",
        "wins": "Wins",
        "losses": "Losses",
        "draws": "Draws",
        "time": "TIME:", 
        "avg_move_time": "Avg Move Time (ms)",
        "settings": "Settings",
        "help": "Help",
        "about_me": "About me",
        "start_game": "Start Game",
        "new_game": "New Game",
        "reset_scores": "Reset Scores",
        "game_result": "Game Result",
        "player_wins": "Player 1 Wins",
        "ai_wins": "Player 2 Wins",
        "reset_scores_warning": "reset scores warning",
        "wins": "Wins",
        "pieces": "Pieces",
        "footer": "",
        "confirm": "Confirm",
        "new_game_warning": "Start a new game?",
        "warning": "Warning",
        "game_settings_tab": "Game",
        "design_tab": "Design",
        "ai_tab": "AI",
        "player_tab": "Players",
        "play_with": "Play With",
        "human_vs_human": "Human vs Human",
        "human_vs_ai": "Human vs AI",
        "ai_vs_ai": "AI vs AI",
        "only_once": "Only Once",
        "repeat_game": "Repeat Game",
        "starting_player": "Starting Player",
        "player": "Player 1",
        "ai": "Player 2",
        "game_timer": "Game Timer",
        "no_timer": "No Timer",
        "with_timer": "With Timer",
        "game_duration": "Duration (min):",
        "language": "Language",
        "color_settings": "Colors",
        "player_piece_color": "Player 1:",
        "ai_piece_color": "Player 2:",
        "board_color_1": "Board Light:",
        "board_color_2": "Board Dark:",
        "piece_style": "Piece Style",
        "sound_settings": "Sound",
        "sound_on": "On",
        "sound_off": "Off",
        "ai_pause_time": "AI Pause Time",
        "ms": "ms",
        "ai_ability": "AI Ability",
        "player_1_name": "Player 1 Name",
        "player_2_name": "Player 2 Name",
        "al1_name": "AL 1 Name",
        "al2_name": "AL2 Name",
        "upload_image": "Upload Image",
        "save_changes": "Save",
        "close": "Close",
        "reset_settings": "Reset",
        "coming_soon": "Coming Soon",
        "player_1_ai_type": "Player 1 AI Type",  # جدید
        "player_2_ai_type": "Player 2 AI Type",  # جدید
        "error": "Error",
        "ai_pause_error": "AI pause time must be between 0 and 5000 ms",
        "invalid_number_hands": "Number of hands must be between 1 and 1000",
        "invalid_number_error": "Please enter valid numbers", "apply_after_game": "Settings will apply after the game ends",
        "reset_settings_warning": "Are you sure you want to reset all settings?"
    },
    "fa": {
        "piece_images": "تصاویر مهره‌ها",
        "player_1_piece": "مهره بازیکن ۱",
        "player_1_king": "شاه بازیکن ۱",
        "player_2_piece": "مهره بازیکن ۲",
        "player_2_king": "شاه بازیکن ۲",
        "piece_images": "تصاویر مهره‌ها",
        "player_1_piece": "مهره بازیکن ۱",
        "player_1_king": "شاه بازیکن ۱",
        "player_2_piece": "مهره بازیکن ۲",
        "player_2_king": "شاه بازیکن ۲",
        "al_progress": "پیشرفت هوش مصنوعی",
        "az_progress":  "پیشرفت هوش مصنوعی",
        "wins": "بردها",
        "losses": "باخت‌ها",
        "draws": "تساوی‌ها",
        "player_1_ai_type": "نوع هوش مصنوعی بازیکن ۱",  # جدید
        "player_2_ai_type": "نوع هوش مصنوعی بازیکن ۲",  # جدید
        "time": "زمان:",
        "avg_move_time": "میانگین زمان حرکت (میلی‌ثانیه)",
        "settings": "تنظیمات",
        "help": "راهنما",
        "about_me": "درباره من",
        "start_game": "شروع بازی",
        "new_game": "بازی جدید",
        "reset_scores_warning": "اخطار بازنشانی امتیازات",
        "reset_scores": "بازنشانی امتیازات",
        "game_result": "نتیجه بازی",
        "player_wins": "بازیکن ۱ برنده شد",
        "ai_wins": "بازیکن ۲ برنده شد",
        "wins": "بردها",
        "pieces": "مهره‌ها",
        "footer": "",
        "confirm": "تأیید",
        "new_game_warning": "آیا بازی جدیدی شروع شود؟",
        "warning": "هشدار",
        "game_settings_tab": "بازی",
        "design_tab": "طراحی",
        "ai_tab": "هوش مصنوعی",
        "player_tab": "بازیکنان",
        "play_with": "بازی با",
        "human_vs_human": "انسان در برابر انسان",
        "human_vs_ai": "انسان در برابر هوش مصنوعی",
        "ai_vs_ai": "هوش مصنوعی در برابر هوش مصنوعی",
        "only_once": "فقط یک بار",
        "repeat_game": "تکرار بازی",
        "starting_player": "بازیکن شروع‌کننده",
        "player": "بازیکن ۱", "ai": "بازیکن ۲",
        "game_timer": "تایمر بازی",
        "no_timer": "بدون تایمر",
        "with_timer": "با تایمر",
        "game_duration": "مدت زمان (دقیقه):",
        "language": "زبان",
        "color_settings": "تنظیمات رنگ",
        "player_piece_color": "بازیکن ۱:",
        "ai_piece_color": "بازیکن ۲:",
        "board_color_1": "روشن صفحه:",
        "board_color_2": "تیره صفحه:",
        "piece_style": "سبک مهره",
        "sound_settings": "صدا",
        "sound_on": "روشن",
        "sound_off": "خاموش",
        "ai_pause_time": "زمان مکث هوش مصنوعی",
        "ms": "میلی‌ثانیه",
        "ai_ability": "توانایی هوش مصنوعی",
        "player_1_name": "نام بازیکن ۱",
        "player_2_name": "نام بازیکن ۲",
        "al1_name": "نام Advanced AI ۱",
        "al2_name": "نام Advanced AI ۲",
        "upload_image": "بارگذاری تصویر",
        "save_changes": "ذخیره",
        "close": "بستن",
        "reset_settings": "بازنشانی",
        "coming_soon": "به زودی",
        "error": "خطا",
        "ai_pause_error": "زمان مکث هوش مصنوعی باید بین ۰ تا ۵۰۰۰ میلی‌ثانیه باشد",
        "invalid_number_hands": "تعداد دست‌ها باید بین ۱ تا ۱۰۰۰ باشد", "invalid_number_error": "لطفاً اعداد معتبر وارد کنید",
        "apply_after_game": "تنظیمات پس از پایان بازی اعمال خواهند شد",
        "reset_settings_warning": "آیا مطمئن هستید که می‌خواهید تمام تنظیمات را بازنشانی کنید؟"
    }
}