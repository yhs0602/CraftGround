import os

from PIL import ImageFont

font = None


def get_font(size: int = 8):
    global font
    if font is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(script_dir, "../DejaVuSans-ExtraLight.ttf")
        font = ImageFont.truetype(font_path, size)
    return font
