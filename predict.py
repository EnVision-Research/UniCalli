# -*- coding: utf-8 -*-
from inference import CalligraphyGenerator

# Initialize generator (using DeepSpeed ZeRO-2)
# Note: Set offload=False when using DeepSpeed, as DeepSpeed manages memory itself
generator = CalligraphyGenerator(
    model_name="flux-dev",
    device="cuda",
    offload=False,  # DeepSpeed manages memory, no need for manual offload
    intern_vlm_path="InternVL3-1B",
    checkpoint_path="unicalli-base_cleaned.bin",
    font_descriptions_path='dataset/chirography.json',
    author_descriptions_path='dataset/calligraphy_styles_en.json',
    use_deepspeed=False,
    use_4bit_quantization=True,
)

# Generate calligraphy (must be 5 characters)
image, cond_img = generator.generate(
    text="生日快乐喵",  # Must be 5 characters
    font_style="草",    # 楷(Regular)/草(Cursive)/行(Running)
    author="黄庭坚",    # Or None to use synthetic style
    save_path="output.png",
    num_steps=25,
    seed=42,
)