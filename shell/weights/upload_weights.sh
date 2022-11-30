if [ "$#" -ne 2 ]; then
  echo USAGE: ./process_backbone_weights.sh WEIGHTS_PATH GCS_PATH
  exit 1
fi

WEIGHTS=$1
GCS_PATH=$2

echo Checksum: $(shasum -a 256 $WEIGHTS)

gsutil cp $WEIGHTS $GCS_PATH/
gsutil acl ch -u AllUsers:R $GCS_PATH/$WEIGHTS
