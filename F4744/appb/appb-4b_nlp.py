import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras_nlp
from keras_nlp import models
print(keras_nlp.__version__)

# ! pip install tensorflow_datasets

import keras
print(keras.__version__)

import numpy as np
import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
model_name = "bert_tiny_en_uncased_sst2"

preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
    model_name,
    sequence_length=128,
)

classifier = keras_nlp.models.BertClassifier.from_preset(
    model_name,
    num_classes=2,
    activation="sigmoid",
    preprocessor=preprocessor
)
classifier.fit(imdb_train, validation_data=imdb_test)

class_names = ["negative", "postive"]
review1 = "What an amazing movie!"
scores = classifier.predict([review1])
print(review1)
print(class_names[np.argmax(scores)], "-", scores)
review2 = "A total waste of my time."
scores = classifier.predict([review2])
print(review2)
print(class_names[np.argmax(scores)], "-", scores)