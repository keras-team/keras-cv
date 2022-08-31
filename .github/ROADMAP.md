# Roadmap
This document provides an overview of where KerasCV is at, and where it is going.
# Ongoing Efforts
## Finish a compelling, end to end, image classification journey
We have strong preprocessing, a sound API for producing classification models, easy to
use data loading utilities, and a plethora of information on what it takes to train
state of the art models.

We need to bundle it all up into a training template that produces a pretrained
checkpoints.  Each model should ideally have a script in `examples` showing how to
produce state of the art results for that model architecture.
The scripts should be easily forkable, use monitoring, automatically checkpoint and
restart from failure, and all in all provide a delightful user experience.

### Visualization Tools

The new `visualization` directory will contain methods to visualize various components of
the training pipeline: i.e. visualize bounding boxes, segmentation maps, and more!

The specific feature set here is still being sorted out.

### Explainability Tools

Explainability tools will show *why* a model makes the classification it does.  This
includes tools like `GradCam`.  A Google Summer of Code student will likely kick this
effort off.

## Provide a unique bounding box journey
Recently we finished our API design for bounding box handling.  With this change,
we now have the opportunity to finish providing strong bounding box support within the
library.

Ideally like to provide the following workflow for training a bounding box detection model:

```python
train_ds = load_dataset()

preprocessing_model = keras.Sequential(
    [
        keras_cv.RandAugment(),
        keras_cv.Mosaic()(),
        keras_cv.MixUp(),
    ], name="preprocessing_model"
)

train_ds = train_ds.map(tfm.RetinaNet.LabelEncoder().encoder_batch())
model = tfm.RetinaNet(keyword_arg_1=my_arv, name="retinanet")

model.compile(
    # we don't need to include loss because it is core to the RetinaNet model
    optimizer=optimizers.SGD(),
    metrics=[
        keras_cv.COCOMeanAveragePrecision(),
        keras_cv.COCORecall()
    ]
)

model.fit(train_ds)

results = model.predict(somedata)
# > results is a dict, {train_targets: train_targets, inference_targets: inference_targets}
```

### Bounding Box Support for Image Data Augmentation Layers
[divyashreepathihalli](https://github.com/divyashreepathihalli) is leading this effort.

As part of v0.1.0, KerasCV launched 28 image data augmentation layers.
These layers provide all that is needed to train state of the art image classification models; namely RandomRotation, ChannelShift, RandAugment, CutMix, and MixUp.
We would like to add support for bounding boxes to these data augmentation techniques.

# Semantic Segmentation

Following bounding box support, another team member will begin focusing on introducing
segmentation map support to the repo.

# Whats Next?

After Semantic Segmentation, what is next for KerasCV?  Following this, we will pick
specific applications such as OCR, or few shot learning, and place a high focus on
providing a seamless end to end journey on completing these tasks.

What tasks will be picked?  This will be decided in the following months.

## Overview
- **KerasCV is not a repository of blackbox end-to-end solutions (like TF Hub or ModelGarden).**

    For the time being, it is focused on modular and reusable building blocks. We expect models in
    ModelGarden or TF Hub to eventually be built on top of the building blocks provided by KerasCV.

    In the process of developing these building blocks, we will by necessity implement end-to-end
    workflows, but they're intended purely for demonstration and grounding purposes, they're not
    our main deliverable.


- **KerasCV is not a repository of low-level image-processing ops, like tf.vision.**

    KerasCV is fundamentally an extension of the Keras API: it hosts Keras objects, like layers,
    metrics, or callbacks. Low-level C++ ops should go elsewhere.


- **KerasCV is a production grade library.**

    Researchers may use it, but we do not consider researchers to be our target audience. Our target
    audience is applied CV engineers with experimentation and production needs. KerasCV should make
    it possible to quickly reimplement industry-strength versions of the latest generation of
    architectures produced by researchers, but we don't expect the research effort itself to be built
    on top of KerasCV. This enables us to focus on usability and API standardization, and produce
    objects that have a longer lifespan than the average research project.

# Community contribution guideline
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
