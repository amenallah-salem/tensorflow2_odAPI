# data & pipeline for object detection using tensorflow object detection API.

### Concept.

### Installation.
```
#go into the project dir 
!cd ./object_detection
and compile protos.
!protoc object_detection/protos/*.proto --python_out=.

```
install TensorFlow 2.x Object Detection API dependencies.
```
!python -m pip install
!pip -r install requirements.txt
```

```

#test the installation with 
!python object_detection/builders/model_builder_tf2_test.py
```
###### docker installation.

### Preparing data for training (Collection and annotation & generating TFRecord format)

convert xml files to csv 
```
!python xml_to_csv.py
```

mofify class names & generate_tfrecords

```
def class_text_to_int(row_label):
    if row_label == 'class1':
        return 1
    elif row_label == 'class2':
        return 2
    else:
        return None

```
to genrate tf records just run the following script:

```!python generate_tfrecord.py \
        --csv_input=images/train_labels.csv \
        --image_dir=images/train \
        --output_path=train.record

!python generate_tfrecord.py \
        --csv_input=images/test_labels.csv \
        --image_dir=images/test \
        --output_path=test.record


```
### training Configuration



Final step: download Pretrained model into the object detection folder 
```
%cd ../training/pretrained_mode
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz 

```
Extract the model
```
!python untar_gz_model.py

```
Finally chose configuration file 

```
!cp object_detection/configs/tf2/MODEL_NAME.config .
```



the output in training folder should be as follow 
```
!ls 
MODEL_NAME.tar.gz
labelmap.pbtxt
ssd_efficientdet_d0_512x512_coco17_tpu-8.config

```
now in the .config file make changes to the number of classes and the nuber of epochs and the type to be detection / not classification along with the necessary paths to train.


### training 
```
!cd /object_detection 

!python model_main_tf2.py \
    --pipeline_config_path=/content/models/research/training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --model_dir=training \
    --alsologtostderr ```
### visualisation and Exporting inference graphs

to visualise all training performances run the tensorboard plateform with the following command 
```
import tensorboard
%tensorboard --logdir=training/train
```

to export inferences and run tests 
```
#test the model 
!python exporter_main_v2.py \
    --trained_checkpoint_dir=training \
    --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config \
    --output_directory inference_graph

```
### Resources
 -Tensorflow Object Detection API Repository
 -Tensorflow Object Detection API Documentation
 -Model Zoo Recommended