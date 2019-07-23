#!/usr/bin/env python3
import os, urllib
import tarfile
import logging
from urllib.request import urlretrieve

###################
# Step0: Global setting
####################
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Constants
IMDB_MLP_MODEL_NAME = "imdb_mlp.model"
IMDB_MLP_MODEL_WEIG = "imdb_mlp.h5"

logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger("IMDBb")
logger.setLevel(logging.INFO)

###################
# Step1: Download IMDB
####################
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = "datas/aclImdb_v1.tar.gz"
dataPath = "datas/aclImdb"
if not os.path.isfile(filepath):
    print("Downloading from {}...".format(url))
    result = urlretrieve(url, filepath)
    print("download: {}".format(result))

if not os.path.isdir(dataPath):
    print("Extracting {} to datas...".format(filepath))
    tfile = tarfile.open(filepath, "r:gz")
    result = tfile.extractall("datas/")

###################
# Step2: Reading IMDB
####################
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re


def rm_tags(text):
    r"""
    Remove HTML markers
    """
    re_tag = re.compile(r"<[^>]+>")
    return re_tag.sub("", text)


def read_files(filetype):
    r"""
    Read data from IMDb folders

    @param filetype(str):
        "train" or "test"

    @return:
        Tuple(List of labels, List of articles)
    """
    file_list = []
    positive_path = os.path.join(os.path.join(dataPath, filetype), "pos")
    for f in os.listdir(positive_path):
        file_list.append(os.path.join(positive_path, f))

    negative_path = os.path.join(os.path.join(dataPath, filetype), "neg")
    for f in os.listdir(negative_path):
        file_list.append(os.path.join(negative_path, f))

    logger.debug("Read {} with {} files...".format(filetype, len(file_list)))
    all_labels = [1] * 12500 + [0] * 12500
    all_texts = []
    for fi in file_list:
        logger.debug("Read {}...".format(fi))
        with open(fi, encoding="utf8") as fh:
            all_texts += [rm_tags(" ".join(fh.readlines()))]

    return all_labels, all_texts


logger.info("Reading training data...")
train_labels, train_text = read_files("train")
logger.info("Reading testing data...")
test_labels, test_text = read_files("test")

###################
# Step3: Tokenize
####################
MAX_LEN_OF_TOKEN = 100

logger.info("Tokenizing document...")
token = Tokenizer(num_words=2000)
""" Create a dictionary of 2,000 words """
token.fit_on_texts(train_text)
""" Read in all training text and select top 2,000 words according to frequency sorting descendingly """

logger.info("Total {} document being handled...".format(token.document_count))
logger.info("Top 10 word index:")
c = 0
for t, i in token.word_index.items():
    print("\t'{}'\t{}".format(t, i))
    c += 1
    if c == 10:
        break
print("")
logger.info("Translating raw text into token number list...")
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

logger.info(
    "Padding/Trimming the token number list to length={}...".format(MAX_LEN_OF_TOKEN)
)
x_train = sequence.pad_sequences(x_train_seq, maxlen=MAX_LEN_OF_TOKEN)
x_test = sequence.pad_sequences(x_test_seq, maxlen=MAX_LEN_OF_TOKEN)

###################
# Step4: Building MODEL
####################
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.models import model_from_json

MODEL_TYPE = "mlp"
IS_RELOAD = False
if MODEL_TYPE == "mlp":
    if os.path.isfile(IMDB_MLP_MODEL_NAME):
        # Reload model
        logger.debug("Reloading model from {}...".format(IMDB_MLP_MODEL_NAME))
        IS_RELOAD = True
        with open(IMDB_MLP_MODEL_NAME, "r") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(IMDB_MLP_MODEL_WEIG)
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
    else:
        model = Sequential()
        model.add(Embedding(output_dim=32, input_dim=2000, input_length=100))
        model.add(Dropout(0.2))
        """Drop 20% neuron during training """
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))
        """ Total 256 neuron in hidden layers"""
        model.add(Dropout(0.35))
        model.add(Dense(units=1, activation="sigmoid"))
        """ Define output layer with 'sigmoid activation' """

logger.info("Model summary:\n{}\n".format(model.summary()))

###################
# Step5: Training
###################
if not IS_RELOAD:
    logger.info("Start training process...")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    train_history = model.fit(
        x_train,
        train_labels,
        batch_size=100,
        epochs=10,
        verbose=2,
        validation_split=0.2,
    )
    print("")
    # Serialized model
    print("\t[Info] Serialized Keras model to %s..." % (IMDB_MLP_MODEL_NAME))
    with open(IMDB_MLP_MODEL_NAME, "w") as f:
        f.write(model.to_json())
    model.save_weights(IMDB_MLP_MODEL_WEIG)

###################
# Step6: Evaluation
###################
logger.info("Start evaluation...")
scores = model.evaluate(x_test, test_labels, verbose=1)
print("")
logger.info("Score={}".format(scores[1]))


predict_classes = model.predict_classes(x_test).reshape(-1)
print("")
sentiDict = {1: "Pos", 0: "Neg"}


def display_test_Sentiment(i):
    r"""
    Show prediction on i'th test data
    """
    logger.debug("{}'th test data:\n{}\n".format(i, test_text[i]))
    logger.info(
        "Ground truth: {}; prediction result: {}".format(
            sentiDict[test_labels[i]], sentiDict[predict_classes[i]]
        )
    )


logger.info("Show prediction on 2'th test data:")
display_test_Sentiment(2)
