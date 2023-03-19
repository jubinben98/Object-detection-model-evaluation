## About the software
The software is made for the evaluation of the model.
Mention the directory where the test images and the labels are present inside the config.ini file which is present in product/etc/

The software uses SSD-MobileNet tensorflow model as default which is present in product/models/.
SSD-mobile net is trained on COCO dataset.

### About the configurations (location -> product/etc/config.ini)
The software uses all the parameters mentioned in the config.ini file.
Following are the definition for the same.

```commandline
1. log_level: Defines the logs which will be displayed during the execution.
   Acceptable parameters = ["INFO", "DEBUG", "ERROR"]

2. model_path: Defines the path where the model is saved.

3. process_images: Defines how many images should the software should read for execution.
                   This parameter is added as sometimes the system may run out of ram due to reading all the images present in the data dirctory.
   Acceptable parameters = ["all", int()]
   if the parameter is set to "all", then the software will read all the images present in the test data directory.
   if number n is specified, then the software will read n number of images from the data directory.

4. cars_iou_th, cars_conf_th, pedestrians_iou_th, pedestrians_conf_th:
   Defines the iou_threshold to be used for selecting the TP cars bounding box predictions.
   
   The number of values you assign to these parameters should be same across all the threshold.
   
   Ex:
        if the requirement is to do the model evaluation using the following combinations of threshold:
        Combination-1:  cars_iou_th=0.5, cars_conf_th=0.7, pedestrians_iou_th=0.8, pedestrians_conf_th=0.9
        Combination-2:  cars_iou_th=0.1, cars_conf_th=0.2, pedestrians_iou_th=0.3, pedestrians_conf_th=0.4
        Combination-3:  cars_iou_th=0.3, cars_conf_th=0.5, pedestrians_iou_th=0.6, pedestrians_conf_th=0.7
        
        In this case the values inside config.ini should be mentioned like this:
        cars_iou_th = 0.5, 0.1, 0.3
        cars_conf_th = 0.7, 0.2, 0.5
        pedestrians_iou_th = 0.8, 0.3, 0.6
        pedestrians_conf_th = 0.9, 0.4, 0.7
        
5. test_images: Defines the path where the images are saved.

6. test_label: Defines the path where the labels (JSON files) are saved.

7. img_height: Defines the images height

8. img_width: Defines the images width
```


## How to run
### 1. Using Docker

#### Step-1. Got to the root directory
```commandline
cd TechnicalAssignment_I/01-Solution-Task-2
```

#### Step-2. Build the docker-compose file
The docker will install all the dependencies and run the software.
```commandline
docker-compose up --build
```
Just make sure to mention the data directory and the threshold values inside the config.ini.

### 2. Manual installation and execution
#### Step-1. Create a virtual-environment
#### Step-2. Go to the following directory
```commandline
cd TechnicalAssignment_I/01-Solution-Task-2/product/docs/
```

#### Step-3. Install the dependencies
```commandline
pip install -r requirements.txt
```

#### Step-4. Go to the directory where the main software is present
```commandline
cd ..
```

#### Step-5. Execute the following command to execute the software
```commandline
python object_detection.py
```


