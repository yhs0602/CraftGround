import os

from PIL import ImageFont

font = None


def get_font():
    global font
    if font is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(script_dir, "../DejaVuSans-ExtraLight.ttf")
        font = ImageFont.truetype(font_path, 8)
    return font
