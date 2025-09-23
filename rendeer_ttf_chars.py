from pathlib import Path
import argparse
import string
from PIL import Image, ImageDraw, ImageFont
try:
    from fontTools.ttLib import TTFont  # type: ignore
except Exception:  # fontTools is optional; we can fallback
    TTFont = None  # type: ignore


def get_cond_data(texts, img_size, font, font_size, font_scale):
    """
    将 texts 中的每个文本放在一个 img_size×img_size 的格子中，按 bbox 居中绘制。
    注意：font_scale 在外部用于设置 font_size 更合适；本函数只做定位与防裁切。
    """
    W = img_size
    H = img_size * len(texts)
    cond_img = Image.new("RGB", (W, H), (255, 255, 255))
    cond_draw = ImageDraw.Draw(cond_img)

    for i, text in enumerate(texts):
        cell_top = i * img_size

        # 1) 在原点测量文本包围盒（top/left 可能为负）
        try:
            l, t, r, b = cond_draw.textbbox((0, 0), text, font=font, anchor="lt")
        except Exception:
            # Pillow 旧版本兜底
            try:
                l, t, r, b = font.getbbox(text)
            except Exception:
                # 最粗暴的兜底：用mask尺寸近似（不太精确，但不至于崩）
                mask = font.getmask(text)
                w, h = mask.size
                l, t, r, b = 0, 0, w, h

        w, h = r - l, b - t

        # 2) 计算让文本 bbox 在当前 cell 内水平/垂直居中的绘制起点
        x = (W - w) // 2 - l
        y = cell_top + (img_size - h) // 2 - t

        # 3) 真正绘制
        cond_draw.text((x, y), text, font=font, fill=(0, 0, 0), anchor="lt")

    return cond_img


def iter_candidate_chars_by_ranges(include_ext_a: bool = True):
    # English letters
    for ch in string.ascii_letters:
        yield ord(ch)

    # CJK Unified Ideographs Extension A (optional)
    if include_ext_a:
        for cp in range(0x3400, 0x4DBF + 1):
            yield cp

    # CJK Unified Ideographs (common)
    for cp in range(0x4E00, 0x9FFF + 1):
        yield cp


def iter_supported_codepoints(font_path: str, include_ext_a: bool = True):
    # If fontTools available, use cmap to filter supported characters
    if TTFont is not None:
        try:
            tt = TTFont(font_path)
            best_cmap = tt.getBestCmap() or {}
            supported_cps = set(best_cmap.keys())
            desired_cps = set(iter_candidate_chars_by_ranges(include_ext_a=include_ext_a))
            for cp in sorted(supported_cps & desired_cps):
                yield cp
            return
        except Exception:
            pass

    # Fallback: just return desired ranges (may include unsupported tofu boxes)
    for cp in iter_candidate_chars_by_ranges(include_ext_a=include_ext_a):
        yield cp


def looks_blank(img: Image.Image) -> bool:
    # Quick check for blank white image
    # Convert to grayscale and check extrema
    gray = img.convert("L")
    extrema = gray.getextrema()
    return extrema == (255, 255)


def main():
    parser = argparse.ArgumentParser(description="Render supported characters in a TTF to 32x32 images.")
    parser.add_argument("--font", default="./unifont/unifont-16.0.04.otf", help="Path to TTF font file")
    parser.add_argument("--out_dir", default="./unifont_chars_128", help="Output directory for images")
    parser.add_argument("--img_size", type=int, default=512, help="Target image size (square)")
    parser.add_argument("--font_scale", type=float, default=0.8, help="Scale of font relative to image size")
    parser.add_argument("--no_ext_a", action="store_true", help="Exclude CJK Extension A range")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    font_path = args.font
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_size = int(args.font_scale * args.img_size)
    font = ImageFont.truetype(font_path, font_size)

    count = 0
    for cp in iter_supported_codepoints(font_path, include_ext_a=not args.no_ext_a):
        ch = chr(cp)
        # Render single character using the same method as get_cond.py
        img = get_cond_data(ch, args.img_size, font, args.img_size, args.font_scale)

        # Basic filter: skip if it's completely blank (e.g., space). This won't filter tofu boxes.
        if looks_blank(img):
            continue

        # Use codepoint-based filename to avoid filesystem issues
        cp_hex = f"U+{cp:04X}"
        out_path = out_dir / f"{cp_hex}.png"

        # Ensure final image is exactly 32x32 (height should already be 32)
        img = img.resize((args.img_size//args.scale, args.img_size//args.scale), Image.NEAREST)

        img.save(out_path)
        count += 1

    print(f"Saved {count} images to {out_dir}")


if __name__ == "__main__":
    main()