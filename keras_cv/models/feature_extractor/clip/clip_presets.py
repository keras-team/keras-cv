"""CLIP presets."""

clip_presets = {
    "clip-vit-base-patch16": {
        "metadata": {
            "description": (
                "The model uses a ViT-B/16 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss. The "
                "model uses a patch size of 16 and input images of size (224, "
                "224)"
            ),
            "params": 149620737,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_pascalvoc/2",
    },
    "clip-vit-base-patch32": {
        "metadata": {
            "description": (
                "The model uses a ViT-B/32 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 32 and input images of size (224, "
                "224)"
            ),
            "params": 151277313,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_pascalvoc/2",
    },
    "clip-vit-large-patch14": {
        "metadata": {
            "description": (
                "The model uses a ViT-L/14 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 14 and input images of size (224, "
                "224)"
            ),
            "params": 427616513,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_pascalvoc/2",
    },
    "clip-vit-large-patch14-336": {
        "metadata": {
            "description": (
                "The model uses a ViT-L/14 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 14 and input images of size (336, "
                "336)"
            ),
            "params": 427944193,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "",
    },
}
