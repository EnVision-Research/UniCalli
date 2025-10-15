# -*- coding: utf-8 -*-
from inference import CalligraphyGenerator

# Initialize generator (using DeepSpeed ZeRO-2)
# Note: Set offload=False when using DeepSpeed, as DeepSpeed manages memory itself
generator = CalligraphyGenerator(
    model_name="flux-dev",
    device="cuda",
    offload=False,  # DeepSpeed manages memory, no need for manual offload
    intern_vlm_path="path/to/InternVL3-1B",
    checkpoint_path="unicalli-base_cleaned.bin",
    font_descriptions_path='dataset/chirography.json',
    author_descriptions_path='dataset/calligraphy_styles_en.json',
    use_deepspeed=True,
    deepspeed_config="ds_config_zero2.json"
)

# Generate calligraphy (must be 5 characters)
image, cond_img = generator.generate(
    text="生日快乐喵",  # Must be 5 characters
    font_style="楷",    # 楷(Regular)/草(Cursive)/行(Running)
    author="赵佶",    # Or None to use synthetic style
    save_path="output2.png",
    num_steps=39,
    seed=1128293374,
    # seed=2249921890
)