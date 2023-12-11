set -e
set -x

cd "${KOKORO_ROOT}/"

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

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
pip install -U pip setuptools

if [ "${KERAS2:-0}" == "1" ]
then
   echo "Keras2 detected."
   pip install -r requirements-common.txt --progress-bar off
   pip install tensorflow~=2.14
   pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu
   pip install torchvision~=0.16.0
   pip install "jax[cpu]"

elif [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off

elif [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off

elif [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off
fi

pip install --no-deps -e "." --progress-bar off

# Run Extra Large Tests for Continuous builds
if [ "${RUN_XLARGE:-0}" == "1" ]
then
   pytest --check_gpu --run_large --run_extra_large --durations 0 \
      keras_cv/bounding_box \
      keras_cv/callbacks \
      keras_cv/losses \
      keras_cv/layers/object_detection \
      keras_cv/layers/preprocessing \
      keras_cv/models/backbones \
      keras_cv/models/classification \
      keras_cv/models/object_detection/retinanet \
      keras_cv/models/object_detection/yolo_v8 \
      keras_cv/models/object_detection_3d \
      keras_cv/models/segmentation \
      keras_cv/models/stable_diffusion
else
   pytest --check_gpu --run_large --durations 0 \
      keras_cv/bounding_box \
      keras_cv/callbacks \
      keras_cv/losses \
      keras_cv/layers/object_detection \
      keras_cv/layers/preprocessing \
      keras_cv/models/backbones \
      keras_cv/models/classification \
      keras_cv/models/object_detection/retinanet \
      keras_cv/models/object_detection/yolo_v8 \
      keras_cv/models/object_detection_3d \
      keras_cv/models/segmentation \
      keras_cv/models/stable_diffusion
fi