if [ "$#" -ne 2 ]; then
  echo USAGE: ./process_backbone_weights.sh WEIGHTS_PATH GCS_PATH
  exit 1
fi

WEIGHTS=$1
GCS_PATH=$2

echo Checksum: $(shasum -a 256 $WEIGHTS)

gcloud storage cp $WEIGHTS $GCS_PATH/
gcloud storage objects update --add-acl-grant=AllUsers:R $GCS_PATH/$WEIGHTS
