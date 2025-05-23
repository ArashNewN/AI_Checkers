import re
import logging

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """تبدیل رنگ هگز به RGB"""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6 or not re.match(r'^[0-9a-fA-F]{6}$', hex_color):
            raise ValueError(f"Invalid hex color: #{hex_color}")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r, g, b
    except ValueError as e:
        logging.warning(f"Invalid hex color: #{hex_color}, error: {str(e)}. Using default color (255, 255, 255)")
        return 255, 255, 255

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """تبدیل RGB به رنگ هگز"""
    try:
        r, g, b = rgb
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise ValueError(f"Invalid RGB values: {rgb}")
        return f"#{r:02x}{g:02x}{b:02x}"
    except ValueError as e:
        logging.warning(f"Invalid RGB input: {rgb}, error: {str(e)}. Using default color #ffffff")
        return "#ffffff"

class CheckersError(Exception):
    """Exception for checkers game errors."""
    pass