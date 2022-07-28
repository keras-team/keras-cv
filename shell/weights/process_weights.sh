WEIGHTS=$1
OUTPUT_WEIGHTS=$2
MODEL=$3
GCS_PATH=$4

python3 remove_top.py --weights_path=$1 --output_weights_path=$2 --model_name=$3 

echo With top: $GCS_PATH$WEIGHTS
echo With top checksum: $(shasum -a 256 $1)
echo Without top: $GCS_PATH$OUTPUT_WEIGHTS
echo Without top checksum: $(shasum -a 256 $2)

gsutil cp $WEIGHTS $GCS_PATH
gsutil cp $OUTPUT_WEIGHTS $GCS_PATH

gsutil acl ch -u AllUsers:R $GCS_PATH/$WEIGHTS
gsutil acl ch -u AllUsers:R $GCS_PATH/$OUTPUT_WEIGHTS
