# Solution for the Ticket Classification case study
Here, if we run the ```predict.py```, it will load the model that we trained and run predictions on the test file. It will classify the tickets into it's respective categories and then it will map those categories to the respective agents.

## Where to get the models from
You can get different glove models from the [stanford nlp website](https://nlp.stanford.edu/projects/glove/), some of the links are here:
* [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB)
* [glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip) Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB)
* [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip) Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB)
* [glove.twitter.27B.zip](https://nlp.stanford.edu/data/glove.twitter.27B.zip) Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB)

Download the model of your choice, unzip it and then select the path of the model version you are going to use, 
whichever version of the model you will use, it will give back the vectors accordingly. If you use the model with 50 vector size, it will return a word vector of size 50. 
There are options for vector size of 50,100,200,300

Here, i have not included the model files in the repo, as that would lead to the repo being bloated. 


