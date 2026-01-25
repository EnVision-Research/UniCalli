# -*- coding: utf-8 -*-
"""
Gradio Demo for Chinese Calligraphy Generation - Hugging Face Spaces Version
"""

import gradio as gr
import spaces
import json
import csv
import torch

# Load author and font mappings from CSV
def load_author_fonts_from_csv(csv_path):
    """
    Load author and their available fonts from CSV file
    Filters out authors that only support 隶 or 篆 fonts
    Returns: dict mapping author to list of font styles
    """
    author_fonts = {}
    excluded_fonts = {'隶', '篆'}  # Fonts we don't support
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row['书法家']
            fonts = row['字体类型'].split('|')  # Split multiple fonts by |
            
            # Filter out unsupported fonts (隶 and 篆)
            supported_fonts = [f for f in fonts if f not in excluded_fonts]
            
            # Only include author if they have at least one supported font
            if supported_fonts:
                author_fonts[author] = supported_fonts
    
    return author_fonts

# Load author-font mappings
AUTHOR_FONTS = load_author_fonts_from_csv('dataset/author_fonts_summary.csv')

# Available authors (sorted)
AUTHOR_LIST = sorted(AUTHOR_FONTS.keys())

# Font style display names (only supported styles)
FONT_STYLE_NAMES = {
    "楷": "楷 (Regular Script)",
    "行": "行 (Running Script)", 
    "草": "草 (Cursive Script)"
}

# Load author descriptions if available
try:
    with open('dataset/calligraphy_styles_en.json', 'r', encoding='utf-8') as f:
        author_styles = json.load(f)
except:
    author_styles = {}

# Global generator (initialized on first use)
generator = None


def init_generator():
    """Initialize the generator"""
    global generator
    if generator is None:
        from inference import CalligraphyGenerator
        
        # On HF Spaces, model will be auto-downloaded from HF Hub
        generator = CalligraphyGenerator(
            model_name="flux-dev",
            device="cuda",
            offload=False,
            intern_vlm_path="OpenGVLab/InternVL3-1B",  # Will be downloaded automatically
            checkpoint_path="unicalli-base_cleaned.bin",  # Will be downloaded from TSXu/UniCalli-base
            font_descriptions_path='dataset/chirography.json',
            author_descriptions_path='dataset/calligraphy_styles_en.json',
            use_deepspeed=False,
            use_4bit_quantization=True,  # Always use 4-bit on ZeroGPU (18G VRAM)
        )
    return generator


def update_font_choices(author: str):
    """
    Update available font choices based on selected author
    """
    if author == "None (Synthetic / 合成风格)" or author not in AUTHOR_FONTS:
        choices = list(FONT_STYLE_NAMES.values())
    else:
        available_fonts = AUTHOR_FONTS[author]
        choices = [FONT_STYLE_NAMES[font] for font in available_fonts if font in FONT_STYLE_NAMES]
    
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


@spaces.GPU(duration=120)
def generate_calligraphy(
    text: str,
    author_dropdown: str,
    font_style: str,
    num_steps: int,
    seed: int,
    random_seed: bool
):
    """
    Generate calligraphy based on user inputs
    """
    # Validate text
    if len(text) != 5:
        raise gr.Error(f"文本必须是5个字符 / Text must be 5 characters. Current: {len(text)}")
    
    # Extract font style value from display name
    font = None
    for font_key, font_display in FONT_STYLE_NAMES.items():
        if font_display == font_style:
            font = font_key
            break
    
    if font is None:
        raise gr.Error(f"无法识别的字体风格 / Unknown font style: {font_style}")
    
    # Determine author
    author = author_dropdown if author_dropdown != "None (Synthetic / 合成风格)" else None
    
    # Handle seed
    if random_seed:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    # Initialize generator if needed
    gen = init_generator()
    
    # Generate
    result_img, cond_img = gen.generate(
        text=text,
        font_style=font,
        author=author,
        num_steps=num_steps,
        seed=seed,
    )
    
    return result_img, f"Seed: {seed}"


# Create Gradio interface
with gr.Blocks(title="UniCalli - Chinese Calligraphy Generator / 中国书法生成器") as demo:
    gr.Markdown("""
    # 🖌️ UniCalli - 中国书法生成器 / Chinese Calligraphy Generator
    
    Generate beautiful Chinese calligraphy in various styles and by different historical masters.
    
    用不同历史书法大师的风格生成精美的中国书法。
    
    **注意 / Note**: 输入文本必须是 **5个汉字** / Input text must be **5 Chinese characters**.
    
    ⚡ Running on ZeroGPU with 4-bit quantization (18G VRAM)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📝 输入设置 / Input Settings")
            
            text_input = gr.Textbox(
                label="输入文本 / Input Text (5个汉字 / 5 characters)",
                placeholder="请输入5个汉字 / Enter 5 Chinese characters, e.g.: 生日快乐喵",
                value="生日快乐喵",
                max_lines=1
            )
            
            gr.Markdown("### 👤 书法家选择 / Calligrapher Selection")
            
            author_dropdown = gr.Dropdown(
                label="1. 选择书法家 / Select Calligrapher",
                choices=["None (Synthetic / 合成风格)"] + AUTHOR_LIST,
                value="黄庭坚",
                info="先选择历史书法家 / Choose a historical calligrapher first"
            )
            
            # Get initial fonts for default author (黄庭坚)
            initial_author = "黄庭坚"
            initial_fonts = AUTHOR_FONTS.get(initial_author, ["草", "行"])
            initial_font_choices = [FONT_STYLE_NAMES[f] for f in initial_fonts if f in FONT_STYLE_NAMES]
            
            font_style = gr.Dropdown(
                label="2. 选择字体风格 / Select Font Style",
                choices=initial_font_choices,
                value="草 (Cursive Script)",
                info="根据所选书法家显示可用字体 / Shows available fonts for selected calligrapher"
            )
            
            gr.Markdown("### ⚙️ 生成设置 / Generation Settings")
            
            num_steps = gr.Slider(
                label="生成步数 / Inference Steps",
                minimum=10,
                maximum=50,
                value=25,
                step=1,
                info="更多步数 = 更高质量，但更慢 / More steps = higher quality, but slower"
            )
            
            with gr.Row():
                seed = gr.Number(
                    label="随机种子 / Seed",
                    value=42,
                    precision=0
                )
                random_seed = gr.Checkbox(
                    label="随机种子 / Random Seed",
                    value=False
                )
            
            generate_btn = gr.Button("🎨 生成书法 / Generate Calligraphy", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 🖼️ 生成结果 / Generated Result")
            gr.Markdown("")  # Add spacing
            
            with gr.Row():
                gr.Column(scale=1)  # Left spacer
                with gr.Column(scale=2):
                    output_image = gr.Image(
                        show_label=False,
                        type="pil",
                        height=600
                    )
                gr.Column(scale=1)  # Right spacer
            
            seed_info = gr.Textbox(
                label="种子信息 / Seed Info",
                interactive=False
            )
    
    # Author info section
    with gr.Accordion("📚 可用书法家列表 / Available Calligraphers（共 {} 位 / {} total）".format(len(AUTHOR_LIST), len(AUTHOR_LIST)), open=False):
        author_info_md = "| 书法家 / Calligrapher | 可用字体 / Available Fonts |\n|--------|----------|\n"
        for author in AUTHOR_LIST[:30]:
            fonts = " | ".join(AUTHOR_FONTS[author])
            author_info_md += f"| **{author}** | {fonts} |\n"
        if len(AUTHOR_LIST) > 30:
            author_info_md += f"\n*... 还有 {len(AUTHOR_LIST) - 30} 位书法家 / {len(AUTHOR_LIST) - 30} more calligraphers*"
        gr.Markdown(author_info_md)
    
    # Event handlers
    author_dropdown.change(
        fn=update_font_choices,
        inputs=[author_dropdown],
        outputs=[font_style]
    )
    
    generate_btn.click(
        fn=generate_calligraphy,
        inputs=[
            text_input,
            author_dropdown,
            font_style,
            num_steps,
            seed,
            random_seed
        ],
        outputs=[output_image, seed_info]
    )
    
    # Examples
    gr.Markdown("### 📋 示例 / Examples")
    gr.Examples(
        examples=[
            ["生日快乐喵", "黄庭坚", "草 (Cursive Script)", 25, 42, False],
        ],
        inputs=[
            text_input,
            author_dropdown,
            font_style,
            num_steps,
            seed,
            random_seed
        ],
    )


if __name__ == "__main__":
    demo.launch()
