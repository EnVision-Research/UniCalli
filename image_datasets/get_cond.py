from PIL import Image, ImageDraw, ImageFont

def get_cond_data(texts, img_size, font, font_size, font_scale):
    img_size = (img_size, img_size * len(texts))
    cond_img = Image.new("RGB", img_size, (255, 255, 255))
    cond_draw = ImageDraw.Draw(cond_img)

    for i, text in enumerate(texts):
        font_space = font_size * (1 - font_scale) // 2
        font_position = (font_space, font_size * i + font_space)
        cond_draw.text(font_position, text, font=font, fill=(0, 0, 0))
    
    return cond_img

img_size = 128
font_scale = 0.8
font_path = "./FangZhengKaiTiFanTi-1.ttf"
font = ImageFont.truetype(font_path, int(font_scale * img_size))
img = get_cond_data(
    "生日快乐喵",
    img_size,
    font,
    img_size,
    font_scale
)

img.save("/data/user/txu647/code/flux-calligraphy/test_data/debug/c.png")
