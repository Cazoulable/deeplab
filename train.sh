VERSION_ID=0
MODEL=deeplabv3_pascal_trainval

PROJECT_DIR=/home/ap/steel/
PATH_TO_INITIAL_CHECKPOINT=$PROJECT_DIR/models/$MODEL/model.ckpt
PATH_TO_TRAIN_DIR=$PROJECT_DIR/experiments/v$VERSION_ID
PATH_TO_DATASET=$PROJECT_DIR/data/tfrecord/

# From $PROJECT_DIR/steel/
python3 deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=100000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
    --output_stride=16 --decoder_output_stride=4 \
    --train_crop_size="256,256" \
    --train_batch_size=20 \
    --dataset="steel" \
    --tf_initial_checkpoint=$PATH_TO_INITIAL_CHECKPOINT \
    --train_logdir=$PATH_TO_TRAIN_DIR \
    --dataset_dir=$PATH_TO_DATASET \
    --initialize_last_layer=False \
    --num_clones=4
