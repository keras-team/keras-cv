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

cd "src/github/keras"
pip install -U pip setuptools

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off
   echo "Check that TensorFlow uses GPU"
   python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("GPU"))'
   # Raise error if GPU is not detected.
   python3 -c 'import tensorflow as tf;assert len(tf.config.list_physical_devices("GPU")) > 0'
fi

if [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   export JAX_ENABLE_X64=true
   pip install -r requirements-jax-cuda.txt --progress-bar off
   python3 -c 'import jax;print(jax.__version__);print(jax.default_backend())'
   # Raise error if GPU is not detected.
   python3 -c 'import jax;assert jax.default_backend().lower() == "gpu"'
fi

if [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off
   python3 -c 'import torch;print(torch.__version__);print(torch.cuda.is_available())'
   # Raise error if GPU is not detected.
   python3 -c 'import torch;assert torch.cuda.is_available()'
fi

pytest --run_large --durations 0 \
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
   keras_cv/models/stable_diffusion \
   --cov=keras-cv
