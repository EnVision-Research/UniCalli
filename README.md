v1: only generation;
v2: general, give 2 timesteps for cond_img and img;
v3: general, provide more detailed text description for fonts and calligraphers;
v3.1: directly output text latent via simple linear;
v3.1-oracle: add oracle and eg data;
v3.2: optimize the text latent prediction by using cross attn.
v4: general, optimize the text prediciton pipeline, using top-k; and, optimize the training tech, (prob<\p) pure noise for condition 