from inspect import Parameter
import tensorflow as tf
from skimage.filters import gaussian
import numpy as np
import cv2
import os
import random
import albumerations as A

class CopyPaste(layers.Layer):
    """
        Basic documentation for the layer
    
    """

    def __init__(self, blend = True, sigma = 1, max_paste_objects = None, *args, **kwargs):
        super().__init__(args, kwargs)
        # Attribute to blend onto the paste image or not
        self.blend = blend
        # Attribute used for gaussian filter calculation
        self.sigma = sigma
        # Attribute containing information about maximum objects to copy paste
        self.max_paste_objects = max_paste_objects

    # Method for performing image copy paste onto the image
    def _image_copy_paste(self, img, alpha, paste_img, blend, sigma):
        if alpha is not None:
            if blend:
                alpha = gaussian(alpha, sigma = sigma, preserve_range = True)
            img_dtype = img.dtype
            alpha = alpha[..., None]
            img = paste_img * alpha + img * (1 - alpha)
            img = img.astype(img_dtype)
        
        return img

    # Method for performing mask copy paste onto the image
    def _mask_copy_paste(self, masks, paste_masks, alpha):
        if alpha is not None:
            masks = [
                np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.unint8) for mask in masks
            ]
            masks.extend(paste_masks)
        
        return masks

    def call(self, images, masks, paste_images, paste_masks):
        """
            Call method for Copy Paste layer
        """
        result_img = images
        result_mask = masks
        for index, image in enumerate(images):
            # A random index for paste image is selected
            selected_index = random.randint(len(paste_images))
            paste_img = paste_img[selected_index]
            # Number of objects to be copy pasted are computed depending on the number of masks present as well as the maximum number of objects (provided by the user)
            if self.max_paste_objects > len(masks[index]):
                n_objects = len(masks[index])
            elif self.max_paste_objects is None:
                n_objects = random.randint(len(masks[index]))
            else:
                n_objects = self.max_paste_objects
            # Selected masks are obtained randomly
            selected_objects = random.sample(masks[index], n_objects)

            # Alpha for the selected masks is computed
            # It is a binary image consisting of all the masks in the image only
            alpha = selected_objects[0] > 0
            for j_mask in selected_objects[1:]:
                alpha += j_mask > 0

            # Copy pasted augmented image as well as masks are obtained using _image_copy_paste and _mask_copy_paste functions
            result_img.append(self._image_copy_paste(image, paste_img, alpha))
            result_mask.append(self._mask_copy_paste(selected_objects, paste_masks[selected_index], alpha))

        # Already existing images and masks provided are appended in the results from previous step to create an augmented dataset
        for image, mask in zip(images, mask):
            result_img.append(image)
            result_mask.append(mask)

        # Augmented images as well as masks are returned from the call function of the layer
        return (result_img, result_mask)