# API Design Guidelines
In general, KerasCV abides to the  [API design guidelines of Keras](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md).

There are a few API guidelines that apply only to KerasCV.  These are discussed
in this document.

## Preprocessing Layers
### Color Based Preprocessing Layers
Some preprocessing layers in KerasCV perform color based transformations.  This
includes `RandomBrightness`, `Equalize`, `Solarization`, and more.  Preprocessing
layers that perform color based transformations make the following assumptions:
- these layers must accept a `value_range`, which is a tuple of numbers.
- `value_range` must default to `(0, 255)`
- input images may be of any `dtype`

Additionally, these preprocessing layers should cast back to the input images
original `dtype` before the end of their `call` body.  This API design decision
is made to preserve simplicity and provide the easiest API to use. 

The decision to support inputs of any `dtype` is made based on the nuance that
some Keras layers cast user inputs without the user knowing.  For example, if
`Solarization` expected user inputs to be of type `int`, and a custom layer
was accidentally casting inputs to `float32`, it would be a bad user experience
to raise an error asserting that all inputs must be of type `int`.  

New preprocessing layers should be consistent with these decisions.
