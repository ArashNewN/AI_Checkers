# utils.py
def hex_to_rgb(hex_str):
    """تبدیل رشته هگزادسیمال (مانند '#ff0000') به تاپل RGB (مانند (255, 0, 0))"""
    if hex_str.startswith('#'):
        hex_str = hex_str[1:]
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """تبدیل تاپل RGB (مانند (255, 0, 0)) به رشته هگزادسیمال (مانند '#ff0000')"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"