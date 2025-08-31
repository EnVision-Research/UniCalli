from PIL import Image, ImageDraw, ImageFont

def render_char(self, ch, bg="white", fg="black"):
    img = None
    for font in self.fonts:
        if ImageFont.FreeTypeFont.getbbox(font, ch) is None:
            continue
        img = Image.new("RGB", self.img_size, bg)
        pad = self.font_size * (1 - self.font_scale) // 2
        ImageDraw.Draw(img).text((pad, pad), ch, font=font, fill=fg)
    return img

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

img.save("/data/user/txu647/code/flux-calligraphy/test_data/ccc.png")
