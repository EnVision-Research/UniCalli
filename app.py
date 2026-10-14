# -*- coding: utf-8 -*-
"""
Gradio Demo for Chinese Calligraphy Generation
"""

import gradio as gr
from inference import CalligraphyGenerator
import json
import csv
import os
from src.flux.runtime import resolve_device

DEVICE = resolve_device(os.environ.get("UNICALLI_DEVICE", "auto"))
SEQUENTIAL_OFFLOAD = DEVICE.type == "mps" and os.environ.get("UNICALLI_SEQUENTIAL_OFFLOAD", "1") == "1"

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

# Initialize generator (will be done lazily on first generation)
generator = None
generator_4bit_state = None  # Track current 4bit quantization state


def init_generator(use_4bit: bool = False):
    """Initialize the generator (lazy loading)"""
    global generator, generator_4bit_state
    
    # Reinitialize if 4bit state changed
    if generator is not None and generator_4bit_state != use_4bit:
        generator = None
    
    if generator is None:
        # Model paths (download via: huggingface-cli download TSXu/UniCalli-base --local-dir ./checkpoints)
        checkpoint_path = "./checkpoints/unicalli-base_cleaned.bin"
        intern_vlm_path = "./checkpoints/internvl_embedding"
        
        generator = CalligraphyGenerator(
            model_name="flux-dev",
            device=DEVICE,
            offload=DEVICE.type == "mps",
            sequential_offload=SEQUENTIAL_OFFLOAD,
            intern_vlm_path=intern_vlm_path,
            checkpoint_path=checkpoint_path,
            font_descriptions_path='dataset/chirography.json',
            author_descriptions_path='dataset/calligraphy_styles_en.json',
            use_deepspeed=False,
            use_4bit_quantization=use_4bit,
            # deepspeed_config="ds_config_zero2.json"
        )
        generator_4bit_state = use_4bit
    return generator


def update_font_choices(author: str):
    """
    Update available font choices based on selected author
    
    Args:
        author: Selected author name
        
    Returns:
        Updated dropdown with available fonts for the author
    """
    if author == "None (Synthetic / 合成风格)" or author not in AUTHOR_FONTS:
        # If no author or synthetic, show all font types
        choices = list(FONT_STYLE_NAMES.values())
    else:
        # Show only fonts available for this author
        available_fonts = AUTHOR_FONTS[author]
        choices = [FONT_STYLE_NAMES[font] for font in available_fonts if font in FONT_STYLE_NAMES]
    
    # Return updated dropdown with first choice as default
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def generate_calligraphy(
    text: str,
    author_dropdown: str,
    font_style: str,
    num_steps: int,
    seed: int,
    random_seed: bool,
    use_4bit: bool
):
    """
    Generate calligraphy based on user inputs
    
    Args:
        text: Input text (must be 5 characters)
        author_dropdown: Selected author from dropdown
        font_style: Selected font style (display name)
        num_steps: Number of denoising steps
        seed: Random seed
        random_seed: Whether to use random seed
        use_4bit: Whether to use 4-bit quantization
    
    Returns:
        Generated image and condition image
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
        import torch
        seed = torch.randint(0, 2**32, (1,)).item()
    
    # Initialize generator if needed (with 4bit setting)
    gen = init_generator(use_4bit=use_4bit)
    
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
with gr.Blocks(title="UniCalli - Chinese Calligraphy Generator / 中国书法生成器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖌️ UniCalli - 中国书法生成器 / Chinese Calligraphy Generator
    
    Generate beautiful Chinese calligraphy in various styles and by different historical masters.
    
    用不同历史书法大师的风格生成精美的中国书法。
    
    **注意 / Note**: 输入文本必须是 **5个汉字** / Input text must be **5 Chinese characters**.
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
                minimum=1 if DEVICE.type == "mps" else 10,
                maximum=100,
                value=8 if DEVICE.type == "mps" else 25,
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
            
            use_4bit = gr.Checkbox(
                label="4bit量化 / 4-bit Quantization",
                value=DEVICE.type == "cuda",
                interactive=DEVICE.type == "cuda",
                info="勾选后VRAM仅需18G，但质量会轻微下降 / Checked: only 18G VRAM needed, slight quality reduction"
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
            desc = author_styles.get(author, "")
            desc_short = desc[:50] + "..." if len(desc) > 50 else desc
            author_info_md += f"| **{author}** | {fonts} |\n"
        if len(AUTHOR_LIST) > 30:
            author_info_md += f"\n*... 还有 {len(AUTHOR_LIST) - 30} 位书法家 / {len(AUTHOR_LIST) - 30} more calligraphers*"
        gr.Markdown(author_info_md)
    
    # Event handlers
    # Update font choices when author changes
    author_dropdown.change(
        fn=update_font_choices,
        inputs=[author_dropdown],
        outputs=[font_style]
    )
    
    # Generate button click
    generate_btn.click(
        fn=generate_calligraphy,
        inputs=[
            text_input,
            author_dropdown,
            font_style,
            num_steps,
            seed,
            random_seed,
            use_4bit
        ],
        outputs=[output_image, seed_info]
    )
    
    # Examples
    gr.Markdown("### 📋 示例 / Examples")
    gr.Examples(
        examples=[
            ["生日快乐喵", "黄庭坚", "草 (Cursive Script)", 8 if DEVICE.type == "mps" else 25, 42, False, DEVICE.type == "cuda"],
        ],
        inputs=[
            text_input,
            author_dropdown,
            font_style,
            num_steps,
            seed,
            random_seed,
            use_4bit
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=55630,
        share=False
    )
