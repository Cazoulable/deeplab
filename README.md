# Steel Project
Kaggle Steel competition : https://www.kaggle.com/c/severstal-steel-defect-detection/overview
Deeplab : https://github.com/tensorflow/models/tree/master/research/deeplab


## Building dataset

First you need to download the data from the competition and organize your folder that way :  
* project  
  * steel (code repo)
    * deeplab
      * ...
    * notebooks
      * ...
  * data  
    * sample_submission.csv  
    * test_images/  
    * train.csv  
    * train_images/  
  
Then, run from the steel repo
```
python3 deeplab/datasets/build_steel_data.py
```   
This scripts will split the training set into train/val sets (default is 90%/10% split) and store the data into tfrecords.
