BIN="$HOME/flsf/cc/bazel-bin/app/data/extract-samples"
TSF_DIR="$HOME/data/tsf_data/kittidata/2011_09_26/"

SAVE_DIR="$HOME/data/feature_learning/"

# Copy bin to tmp
TMP_BIN="/tmp/flow_data"
if [ -f $TMP_BIN ];
then
  rm $TMP_BIN -f
fi

cp $BIN $TMP_BIN

for log in $TSF_DIR/2011*
do
  fn=$(basename $log)
  log_num="${fn#*drive_}"
  log_num="${log_num%%_*}"
  echo $log_num

  # Run extractor
  if [ ! -f $SAVE_DIR/$fn/matches.bin ];
  then
    echo "Match file not found, generating data"

    mkdir -p $SAVE_DIR/$fn

    OUT_FILE=$SAVE_DIR/$fn/out.txt

    echo $TMP_BIN --log-num $log_num --save-path $SAVE_DIR/$fn
    $TMP_BIN --log-num $log_num --save-path $SAVE_DIR/$fn >> $OUT_FILE
  fi
done
