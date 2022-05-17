# ------------------------------------------------------------------------ #
import pickle
import pandas as pd

# ------------------------------------------------------------------------ #

# Loading the models that we saved
with open('./models/models.pkl', 'rb') as f:
    cnt_vectorizer, log_clf, glove_clf, labels_encoder = pickle.load(f)

# ------------------------------------------------------------------------ #

# Reading the test data
test_data = pd.read_csv('./test_data/test.csv')

# ------------------------------------------------------------------------ #

# Converting the test data into count vectors
count_vectors = cnt_vectorizer.transform(test_data['0'])

# ------------------------------------------------------------------------ #

# Converting the test data into glove vectors
import numpy as np

# Function to load the glove model
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

glove_model = load_glove_model('./models/glove.6B/glove.6B.100d.txt')

# Function to return glove vector for a sentence
def return_glove_vector(text):
    vectors = []
    text_list = text.split(' ')
    for i in text_list:
        try:
            vectors.append(glove_model[str(i)])
        except:
            pass
        
    np_vectors = np.array(vectors)
    np_vectors_single = np_vectors.mean(axis=0)
    np_vectors_single = np_vectors_single.tolist()
    return np_vectors_single

glove_features_train_transformed =  test_data['0'].apply(return_glove_vector)
glove_features_train_transformed_list = glove_features_train_transformed.to_list()

# ------------------------------------------------------------------------ #


print('---------------------- Count vectorizer results ----------------------')
# Making the prediction using our loaded logistic model
prediction_labels = log_clf.predict(count_vectors)
predictions = labels_encoder.inverse_transform(prediction_labels)
print('Our predicted labels are')
print('\n')
print(predictions)
print('\n')

# Making the dictionary that will tell, who should respond
respondent = pd.read_csv('./test_data/respondent.csv')
respondent.set_index('issue_id',inplace=True)
respondent_dict = respondent.to_dict().get('respondent_id')

# Telling, who should reply to that complaint
respondents_list = [respondent_dict[k] for k in predictions]

print('And the people who should respond are')
print('\n')
print(respondents_list)
print('\n')

print('---------------------- END ----------------------')
print('\n')

# ------------------------------------------------------------------------ #

print('---------------------- Glove vectorizer results ----------------------')

# Making the prediction using our Glove logistic model
prediction_labels = glove_clf.predict(glove_features_train_transformed_list)
predictions = labels_encoder.inverse_transform(prediction_labels)
print('Our predicted labels are')
print('\n')
print(predictions)
print('\n')

# Telling, who should reply to that complaint
respondents_list = [respondent_dict[k] for k in predictions]

print('And the people who should respond are')
print('\n')
print(respondents_list)
print('\n')

print('---------------------- END ----------------------')
print('\n')

# ------------------------------------------------------------------------ #
