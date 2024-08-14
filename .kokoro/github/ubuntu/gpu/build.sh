set -e
set -x

cd "${KOKORO_ROOT}/"

PYTHON_BINARY="/usr/bin/python3.9"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:"
# Check cuda
nvidia-smi
nvcc --version

cd "src/github/keras-cv"
pip install -U pip setuptools psutil

if [ "${KERAS2:-0}" == "1" ]
then
   echo "Keras2 detected."
   pip install -r requirements-common.txt --progress-bar off --timeout 1000
   pip install tensorflow~=2.15.0
   pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu
   pip install torchvision~=0.16.0
   pip install "jax[cpu]"
   pip install keras-nlp-nightly --no-deps
   pip install tensorflow-text==2.15.0

elif [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off --timeout 1000
   pip install keras-nlp-nightly --no-deps
   pip install tensorflow-text~=2.16.0

elif [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off --timeout 1000
   pip install keras-nlp-nightly --no-deps
   pip install tensorflow-text~=2.16.0

elif [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off --timeout 1000
   pip install keras-nlp-nightly --no-deps
   pip install tensorflow-text~=2.16.0
fi

pip install --no-deps -e "." --progress-bar off

# Run Extra Large Tests for Continuous builds
if [ "${RUN_XLARGE:-0}" == "1" ]
then
   pytest --cache-clear --check_gpu --run_large --run_extra_large --durations 0 \
      keras_cv/src/bounding_box \
      keras_cv/src/callbacks \
      keras_cv/src/losses \
      keras_cv/src/layers/object_detection \
      keras_cv/src/layers/preprocessing \
      keras_cv/src/models/backbones \
      keras_cv/src/models/classification \
      keras_cv/src/models/object_detection/retinanet \
      keras_cv/src/models/object_detection/yolo_v8 \
      keras_cv/src/models/object_detection/faster_rcnn \
      keras_cv/src/models/object_detection_3d \
      keras_cv/src/models/segmentation \
      keras_cv/src/models/feature_extractor/clip \
      keras_cv/src/models/stable_diffusion
else
   pytest --cache-clear --check_gpu --run_large --durations 0 \
      keras_cv/src/bounding_box \
      keras_cv/src/callbacks \
      keras_cv/src/losses \
      keras_cv/src/layers/object_detection \
      keras_cv/src/layers/preprocessing \
      keras_cv/src/models/backbones \
      keras_cv/src/models/classification \
      keras_cv/src/models/object_detection/retinanet \
      keras_cv/src/models/object_detection/yolo_v8 \
      keras_cv/src/models/object_detection/faster_rcnn \
      keras_cv/src/models/object_detection_3d \
      keras_cv/src/models/segmentation \
      keras_cv/src/models/feature_extractor/clip \
      keras_cv/src/models/stable_diffusion
fi