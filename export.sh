VERSION_ID=0

MODEL=xception_65

PROJECT_DIR=/home/ap/steel
PATH_TO_TRAIN_DIR=$PROJECT_DIR/experiments/v$VERSION_ID/model.ckpt-47689
PATH_TO_EXPORT_DIR=$PROJECT_DIR/experiments/export/$MODEL/v$VERSION_ID
mkdir -p $PATH_TO_EXPORT_DIR

python3 deeplab/export_model.py \
    --checkpoint_path=$PATH_TO_TRAIN_DIR \
    --export_path=$PATH_TO_EXPORT_DIR/frozen_inference_graph.pb \
    --num_classes=5 \
    --crop_size="256,1600" \
    --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
    --output_stride=16 --decoder_output_stride=4 \
    --inference_scales=1.0 \
    --model_variant=$MODEL \
    --save_inference_graph=False

tar -czvf $PROJECT_DIR/export/$MODEL-v$VERSION_ID.tar.gz $PATH_TO_EXPORT_DIR

