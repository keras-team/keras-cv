# Call For Contributions
Contributors looking for a task can look at the following list to find an item
to work on.  Should you decide to contribute a component, please comment on the 
corresponding GitHub issue that you will be working on the component.  A team 
member will then follow up by assigning the issue to you.

## Default Parameters
Default parameters should **not** be set on which values achieve the best scores
on any specific dataset.  Instead, parameters should be required, with a 
recommended value in the docstring.  This is to discourage users from widely 
adopting the hyperparameters required to do well on ImageNet2k on datasets
that require a different tuning.

## Preprocessing Layers
KerasCV preprocessing layers allow for construction of state of the art computer
vision data augmentation pipelines.  Our [CutMix](https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing/cut_mix.py) implementation serves as a sample preprocessing
layer.

Currently, we are looking for contributions of the following layers:
- [SLIC Layer](https://github.com/keras-team/keras-cv/issues/43)
- [GridMask Layer](https://github.com/keras-team/keras-cv/issues/31)

## Model Architectures
For now, we are not actively seeking contributions of this type.  Once
the KerasCV authors contribute an example of this class of component, we will 
open up the patth for community contributions.

## Visualization Tools
For now, we are not actively seeking contributions of this type.  Once
the KerasCV authors contribute an example of this class of component, we will 
open up the patth for community contributions.
