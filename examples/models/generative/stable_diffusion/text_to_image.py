"""
Title: Generate an image from a text prompt using StableDiffusion
Author: fchollet
Date created: 2022/09/24
Last modified: 2022/09/24
Description: Use StableDiffusion to generate an image according to a short text
             description.
"""

from PIL import Image

from keras_cv.models import StableDiffusion

model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
img = model.text_to_image(
    "Photograph of a beautiful horse running through a field"
)
Image.fromarray(img[0]).save("horse.png")
print("Saved at horse.png")
