# Roadmap
## Non-Goals
- **KerasCV is not a repository of blackbox end-to-end solutions (like TF Hub or ModelGarden).**

    For the time being, it is focused on modular and reusable building blocks. We expect models in 
    ModelGarden or TF Hub to eventually be built on top of the building blocks provided by KerasCV.

    In the process of developing these building blocks, we will by necessity implement end-to-end 
    workflows, but they're intended purely for demonstration and grounding purposes, they're not 
    our main deliverable.


- **KerasCV is not a repository of low-level image-processing ops, like tf.vision.** 

    KerasCV is fundamentally an extension of the Keras API: it hosts Keras objects, like layers, 
    metrics, or callbacks. Low-level C++ ops should go elsewhere.


- **KerasCV is not a research library.**

    Researchers may use it, but we do not consider researchers to be our target audience. Our target 
    audience is applied CV engineers with experimentation and production needs. KerasCV should make 
    it possible to quickly reimplement industry-strength versions of the latest generation of 
    architectures produced by researchers, but we don't expect the research effort itself to be built
    on top of KerasCV. This enables us to focus on usability and API standardization, and produce 
    objects that have a longer lifespan than the average research project.

## Areas of focus
At the start of the project, we would like to take a task-oriented approach to outline our project, 
and the first 2 tasks we would like to focus are:

1. **Image classification** (Given an image and a list of labels, return the label and confidence for 
the image).
2. **Object detection** (Given an image and a list of labels, return a list of bounding box, 
corresponding labels and confidence for the objects in the image).

For each of the tasks, we are targeting to provide user model architectures, layers, preprocessing
functions, losses, metrics and other necessary components to achieve state of the art accuracy.

There will be common shareable components between those 2 tasks, like model building blocks, etc. We
will have them in the repository as public API as well.

### Image Classification
- Dataset to evaluate
    - [Imagenet](https://www.tensorflow.org/datasets/catalog/imagenet2012)
    - [Imagenet v2](https://www.tensorflow.org/datasets/catalog/imagenet_v2) (for testing)
- Target Metric 
    - Top1 Accuracy > 80% (Note that some existing SotA models might not achieve this, like 
original ResNet, but this serve as a guideline for newly added models)

### Object Detection
- Dataset to evaluate
    - [Coco 2017](https://www.tensorflow.org/datasets/catalog/coco#coco2017)
    - [Open image v4](https://www.tensorflow.org/datasets/catalog/open_images_v4)
    - [PASCAL voc](https://www.tensorflow.org/datasets/catalog/voc)
- Target Metric
    - COCO metric mean average precision (mAP)

## Overlapping between keras.application and Keras-CV
As you might notice, some area we would like to focus has overlapping with existing keras.application,
and we would like to clarify the reason of the potential code duplication here.

keras.application has a strong API contract that we can't easily make any backward incompatible
change. On the other hand, there are a few of behavior we actually would like to change within the keras-cv
- Fetch imagenet weights by default, which might not want to do for keras-cv.
- Have the image preprocessing logic outside the model. We want to include all the preprocessing
within the model, so that user won't need to call the extra preprocessing methods.
- Only end-to-end model are exposed as public API, and Keras CV would like to include more building 
blocks as well.
- Some legacy model architectures are no longer commonly used and can't reach SotA performance, which
we might want to deprecate.

Once the keras-cv applications are mature enough, we would add a deprecation warning in the 
keras.applications code.


## Community contribution guideline
We would like to leverage/outsource the Keras community not only for bug reporting or fixes, 
but also for active development for feature delivery. To achieve this, we will have the predefined
process for how to contribute to this repository.

The ideal workflow will be:

- Keras defines the API interface for a certain task, eg a layer, model, dataset, metrics, etc. 
- Post it to Github as a feature request. 
- Have a hotlist/tag for those items and advocate them to the community.
- Let community members claim the task they would like to work on.

User requested features will be carefully evaluated by keras team member before acceptance. 
Based on the existing experience from tf-addons, even with a guideline of paper ref count > 50, 
the image/cv related contributions are still quite scattered. We will make sure the contributions
are coherent with the tasks/themes we have in the repository, or widely used for CV tasks. 
