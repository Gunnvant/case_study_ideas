## Detecting Tags of Programming Questions

### Business Objective
A newage ed-tech provider is building a new feature in their LMS that can help students connect with the right set of people to answer their programming questions. They are looking to build an AI driven feature to detect the programming language from the question posted by students.

### Problem Statement
You have been given a labelled dataset consisting of programming questions asked and their corresponding labels.

### Data Description
The dataset provided for this `45000` annotated examples. This dataset is in a flat file. You can download the dataset from [here](https://drive.google.com/file/d/1IQZZBpBPt9kVDUpuwFkyIN7EPFp62B-k/view?usp=share_link).

### Data Exploration and Preparation
- You will need to pre-process the data before you start building the model
- Process column `Tags`, this will eventually become the target variable for model training. Instead of trying to predict all the tags, you can modify the problem and instead predict only one tag per question. Hint: You can select the first tag in case there are more than one tags present
- Find out how many unique tags are present once you've processed the `Tags` column
-  Once you are satisfied with the eda process, put your code in a file called `eda.py`
- This file should be able to do the following:
    - Read the raw data file
    - Process the tags column
    - Count the unique number of tags (after preprocessing)
    - Print the tag count
    - Divide the data into train and test parts
    - Write the pre-processed files to a location on file system

### Modelling 
- Create file called, `model.py` which will have the definition of the model(s) you wish to use for training. You can try training RNN/LSTMS or traditional text classifiers with tfidf features
- Create a file called, `config.json`, this should contain the hyperparameters such as learning rate, number of epochs, model to be used (defined in `model.py`)
- Create a file called `training.py`, this should import the model from `model.py` and training configuration from `config.json`. This should be able to:
    - Save the model with the following naming convention `model<timestamp>.h5`
    - Write model metrics such as accuracy,f1 score and values of hyperparameters used into a text file called `metrics.txt`

### Inference
- Create a model inference file that can
    - Read a raw text input (Programming Question)
    - Run the pre-processing steps necessary to do model prediction
    - Produce prediction and return the predicted label as well as inference time. This should produce an output as given below:
    ```json
        {
            "label":"C#",
            "inference_time":"0.2 seconds"
        }
    ```