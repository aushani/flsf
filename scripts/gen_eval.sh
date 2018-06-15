BIN="~/flsf/cc/bazel-bin/app/evaluation/evaluate"
TSF_DIR="~/data/tsf_data/kittidata/2011_09_26/"

SAVE_DIR="~/data/eval_flsf/"

# Copy bin to tmp
TMP_BIN="/tmp/flow"
if [ -f $TMP_BIN ];
then
  rm $TMP_BIN -f
fi

cp $BIN $TMP_BIN

SMOOTHING=0.07

for log in $TSF_DIR/2011*
do
  fn=$(basename $log)
  log_num="${fn#*drive_}"
  log_num="${log_num%%_*}"
  echo $log_num

  OUT_PATH=$SAVE_DIR/$fn/$smoothing

  # Run extractor
  if [ ! -f $OUT_PATH/out.txt ];
  then
    echo "Eval file not found, generating data"

    mkdir -p $OUT_PATH

    OUT_FILE=$OUT_PATH/out.txt

    echo $TMP_BIN --log-num $log_num --save-path $OUT_PATH --smoothing $SMOOTHING
    $TMP_BIN --log-num $log_num --save-path $OUT_PATH --smoothing $SMOOTHING >> $OUT_FILE
  else
    echo "Evaluation exists"
  fi
done
