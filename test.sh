VERSION_ID=0
MODEL=deeplabv3_pascal_trainval

PROJECT_DIR=/home/ap/steel
PATH_TO_INITIAL_CHECKPOINT=$PROJECT_DIR/models/$MODEL/model.ckpt
PATH_TO_TRAIN_DIR=$PROJECT_DIR/experiments/v$VERSION_ID/model.ckpt-47689
PATH_TO_VAL_DIR=$PROJECT_DIR/experiments/evals/
PATH_TO_DATASET=$PROJECT_DIR/data/tfrecord/

python3 deeplab/eval.py \
    --checkpoint_dir=$PATH_TO_TRAIN_DIR \
    --eval_logdir=$PATH_TO_VAL_DIR \
    --eval_batch_size=2 \
    --eval_crop_size="256,1600" \
    --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
    --output_stride=16 --decoder_output_stride=4 \
    --eval_scales=1.0 \
    --dataset="steel" \
    --dataset_dir=$PATH_TO_DATASET \
    --model_variant="xception_65" \

