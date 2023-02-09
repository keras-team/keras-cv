"""
Title: Editing generated images with Prompt-to-Prompt
Author: miguelcalado
Date created: 2023/02/09
Last modified: 2023/02/09
Description: Use Prompt-to-Prompt methods to edit and improve StableDiffusion generated images.
"""
import tensorflow as tf
from PIL import Image

from keras_cv.models.stable_diffusion import StableDiffusion

# Recommendation: if you have a low memory gpu drop the batch to 1
BATCH_SIZE = 2
NUM_STEPS = 50
UNCONDITIONAL_GUIDANCE_SCALE = 8

# Stable Diffusion 1.x
generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,
)

# Lets start by generating some chiwawas
print("Generating pictures of chiwawas")
prompt = "a photo of a chiwawa with sunglasses"
seed = 1235
img_org = generator.text_to_image(
    prompt=prompt,
    num_steps=NUM_STEPS,
    unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
    seed=seed,
    batch_size=BATCH_SIZE,
)

# Save the chiwawas
for i in range(len(img_org)):
    filename = f"orig_chiwawa_{i}.png"
    Image.fromarray(img_org[i]).save(filename)
    print(f"Saved at {filename}")

# Method 1: Word swap - The user swaps a single token in the original prompt
list_images_edit = []
SWAPS_WORDS = ["cat", "raccoon", "glasses", "goggles"]
# Variables that control the attention map injections
# To obtain the desired results you need to tinker with them a bit
self_attn_steps = 0.4  # The authors recommend between 20%-40%
cross_attn_steps = 0.6

print("\nMethod 1: Word swap")
for i, swap_word in enumerate(SWAPS_WORDS):
    if i < 2:
        prompt_edit = f"a photo of a {swap_word} with sunglasses"
    else:
        prompt_edit = f"a photo of a chiwawa with {swap_word}"

    # Clean up the session to avoid clutter from old models and layers
    tf.keras.backend.clear_session()
    # Generate Prompt-to-Prompt
    img_edit = generator.text_to_image_prompt_to_prompt(
        prompt="a photo of a chiwawa with sunglasses",
        prompt_edit=prompt_edit,
        method="replace",
        self_attn_steps=self_attn_steps,
        cross_attn_steps=cross_attn_steps,
        num_steps=NUM_STEPS,
        unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
        seed=seed,
        batch_size=BATCH_SIZE,
    )
    list_images_edit.append(img_edit)

# Save edited images
for images_edit, swap_word in zip(list_images_edit, SWAPS_WORDS):
    for i in range(len(images_edit)):
        filename = f"replace_chiwawa_{swap_word}_{i}.png"
        Image.fromarray(images_edit[i]).save(filename)
        print(f"Saved at {filename}")

# Method 2: Prompt refinement - The user adds or replaces new tokens to the prompt.
list_images_edit = []
list_prompt_edit = [
    "a photo of a chiwawa with heart shaped sunglasses",
    "a photo of a chiwawa with aviator sunglasses",
]
# Variables that control the attention map injections
# To obtain the desired results you need to tinker with them a bit
self_attn_steps = 0.2
cross_attn_steps = 0.6

print("Method 2: Prompt refinement")
for prompt_edit in list_prompt_edit:
    # Clean up the session to avoid clutter from old models and layers
    tf.keras.backend.clear_session()
    # Generate Prompt-to-Prompt
    img_edit = generator.text_to_image_prompt_to_prompt(
        prompt="a photo of a chiwawa with sunglasses",
        prompt_edit=prompt_edit,
        method="refine",
        self_attn_steps=self_attn_steps,
        cross_attn_steps=cross_attn_steps,
        num_steps=NUM_STEPS,
        unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
        seed=seed,
        batch_size=BATCH_SIZE,
    )
    list_images_edit.append(img_edit)

# Save edited images
for j, images_edit in enumerate(list_images_edit):
    for i in range(len(images_edit)):
        filename = f"refine_chiwawa_prompt_{j}_{i}.png"
        Image.fromarray(images_edit[i]).save(filename)
        print(f"Saved at {filename}")

# Method 3: Attention Re-weight - the user attributes weights to certain tokens in the prompt,
# strengthening or weakening the effect that the targeted tokens have on the generated output.

# Lets generate some teddy bears
prompt = "a fluffy teddy bear"
seed = 123456

# Clean up the session to avoid clutter from old models and layers
print("Method 3: Attention Re-weight\n Generating fluffy teddy bears")
tf.keras.backend.clear_session()
img_org = generator.text_to_image(
    prompt=prompt,
    num_steps=NUM_STEPS,
    unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
    seed=seed,
    batch_size=1,
)

# Save image
Image.fromarray(img_org[0]).save("teddy_bear.png")

# We need to create the attention weights that we want to manipulate
# For example, lets generate a less stuffed and fluffy teddy bear
prompt_weights = [("fluffy", -5)]
# This creates an array where the token corresponding to the work "fluffy" has weight -5
attn_weights = generator.create_attn_weights(prompt, prompt_weights)

self_attn_steps = 0.2
attn_edit_weights = 0.6

# Clean up the session to avoid clutter from old models and layers
tf.keras.backend.clear_session()
# Generate Prompt-to-Prompt
img_edit = generator.text_to_image_prompt_to_prompt(
    prompt=prompt,
    prompt_edit=prompt,
    method="reweight",
    self_attn_steps=self_attn_steps,
    cross_attn_steps=attn_edit_weights,
    attn_edit_weights=attn_weights,
    num_steps=NUM_STEPS,
    unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
    seed=seed,
    batch_size=1,
)

Image.fromarray(img_edit[0]).save("teddy_bear_less_fluffy.png")

# Or make the bear more "fluffy"
prompt_weights = [("fluffy", 3)]
attn_weights = generator.create_attn_weights(prompt, prompt_weights)

# Clean up the session to avoid clutter from old models and layers
tf.keras.backend.clear_session()

# Generate Prompt-to-Prompt
img_edit = generator.text_to_image_prompt_to_prompt(
    prompt=prompt,
    prompt_edit=prompt,
    method="reweight",
    self_attn_steps=self_attn_steps,
    cross_attn_steps=attn_edit_weights,
    attn_edit_weights=attn_weights,
    num_steps=NUM_STEPS,
    unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
    seed=seed,
    batch_size=1,
)

Image.fromarray(img_edit[0]).save("teddy_bear_more_fluffy.png")
