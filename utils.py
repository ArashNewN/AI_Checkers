def hex_to_rgb(hex_str):
    """تبدیل رشته هگزادسیمال (مانند '#ff0000') به تاپل RGB (مانند (255, 0, 0))

    Args:
        hex_str (str): رشته هگزادسیمال با یا بدون '#'

    Returns:
        tuple: تاپل RGB (مانند (255, 0, 0))

    Raises:
        ValueError: اگر رشته هگزادسیمال نامعتبر باشد
    """
    if not isinstance(hex_str, str):
        raise ValueError("Input must be a string")
    if hex_str.startswith('#'):
        hex_str = hex_str[1:]
    if len(hex_str) != 6 or not all(c in '0123456789abcdefABCDEF' for c in hex_str):
        raise ValueError("Invalid hexadecimal color string")
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError("Invalid hexadecimal color string")

def rgb_to_hex(rgb):
    """تبدیل تاپل RGB (مانند (255, 0, 0)) به رشته هگزادسیمال (مانند '#ff0000')

    Args:
        rgb (tuple): تاپل ۳ تایی از اعداد ۰ تا ۲۵۵

    Returns:
        str: رشته هگزادسیمال (مانند '#ff0000')

    Raises:
        ValueError: اگر تاپل RGB نامعتبر باشد
    """
    if not isinstance(rgb, (tuple, list)) or len(rgb) != 3:
        raise ValueError("Input must be a tuple or list with 3 elements")
    if not all(isinstance(v, int) and 0 <= v <= 255 for v in rgb):
        raise ValueError("RGB values must be integers between 0 and 255")
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"