# Steel Project
Kaggle Steel competition : https://www.kaggle.com/c/severstal-steel-defect-detection/overview
Deeplab : https://github.com/tensorflow/models/tree/master/research/deeplab


## Building dataset

First you need to download the data from the competition and organize your folder that way :  
```
+ project
  + steel (code repo)
    + deeplab
      + ...
    + notebooks
      + ...
  + data
    + sample_submission.csv  
    + test_images/  
    + train.csv  
    + train_images/  
```
Then, run from the steel repo
```
python3 deeplab/datasets/build_steel_data.py
```   
This scripts will split the training set into train/val sets (default is 90%/10% split) and store the data into tfrecords.


## Training


A local training job using `xception_65` can be run with the following command:

```bash
# From tensorflow/models/research/
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```
