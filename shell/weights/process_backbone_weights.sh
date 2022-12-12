if [ "$#" -ne 4 ]; then
  echo USAGE: ./process_weights.sh WEIGHTS_PATH OUTPUT_WEIGHTS_PATH MODEL_NAME GCS_PATH
  exit 1
fi

WEIGHTS=$1
OUTPUT_WEIGHTS=$2
MODEL=$3
GCS_PATH=$4

python3 remove_top.py --weights_path=$WEIGHTS --output_weights_path=$OUTPUT_WEIGHTS --model_name=$MODEL

echo With top: $GCS_PATH/$WEIGHTS
echo With top checksum: $(shasum -a 256 $WEIGHTS)
echo Without top: $GCS_PATH/$OUTPUT_WEIGHTS
echo Without top checksum: $(shasum -a 256 $OUTPUT_WEIGHTS)

gsutil cp $WEIGHTS $GCS_PATH/
gsutil cp $OUTPUT_WEIGHTS $GCS_PATH/

gsutil acl ch -u AllUsers:R $GCS_PATH/$WEIGHTS
gsutil acl ch -u AllUsers:R $GCS_PATH/$OUTPUT_WEIGHTS
