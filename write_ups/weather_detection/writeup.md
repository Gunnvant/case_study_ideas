## Detecting weather

### Business Objective
A windmill company is looking out for a solution that will automate the process of doing maintenance based on the weather conditions. They are looking for a image classifier that can detect the weather conditions and respond appropriately.

### Problem Statement

You have been assigned the task of building an Image classification model which can predict the weather conditions based on input images

### Data Description
The dataset provided for this activity consists of `1531` images belonging to `5` classes. You can find the dataset [here](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset). There are also test images in this dataset on which you are supposed to run model predictions.

### Model Building

- Do an exploratory data analysis and find out how many images are in each class. Also find out the distribution of image dimensions across all the classes.
- Decide the appropriate resizing strategy for the images based on the analysis done above
- You will need to organize your code in the following python scripts:
    - `eda.py`: This should report frequency distribution by each class as well as size distribution of the images. This should produce and save appropriate report files or images on the disk.
    - `train_test_split`: Use this script to create train, validation and test folders from and put 70% of the images in the train folder, 15% in validation and 15% in the test folder. Each of the train, validation and test folders should have the following structure:
    ```
        - Train
            - cloudy
            - foggy
            - rainy
            - shine
            - sunrise
    ```
    - `models.py`: Model definitions should be stored in this file. You will need to define atleast two models. 
        1. Custom CNN: You are free to choose its architecture in terms of layers, activation functions etc
        2. CNN backed by a pre-trained model: You can use vgg16, resnet50, mobilnet etc to define a model that will be used for transfer learning
    - `config.py`: This file should contain hyperparameter of the models such as (but not limited to):
        1. Learning rate
        2. Epochs
        3. Path of train, validation and test data
        4. Type of optimizer used
        5. Type of model used (Custom CNN, CNN with pre-trained model)
    - `train.py`: This should contain the logic to run model training loop. This should produce a trained model as an output and should save the model to disk. You can *optionally* also implement logging of model metrics and saving best model checkpoints.
    - `infer.py`: This should contain the logic to run predictions on test images. This should produce a json file with the following structure:
    ```json
    {
        "predictions":[
            "image_name1":"prediction1",
            "image_name2":"prediction2",
            ...
        ]
    }
    ```