import os
import sys
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split

df=pd.read_csv('train.tsv', sep='\t')
x_train, x_test, y_train, y_test = train_test_split(df['text'],df['label'], test_size=0.2,random_state=42)

maxlen = 128
truncating = 'post'
padding = 'post'
oov_tok = '<00V>'
vocab_size = 1000

tokenizer = Tokenizer(num_words = vocab_size, char_level = False, oov_token = oov_tok)
tokenizer.fit_on_texts(x_train.astype("str"))

# Load model
model = load_model('./sentinel.keras')

# Constants
WELCOME = "Welcome to Sentinel, this bot will help provide insight to the probability of a statement being true. Please enter your statement below.\n"
STATEMENT = "Enter: "
RESULT = "According to our prediction model, that statement is most likely "
FOLLOWUP = 'To Continue, please press "c + Enter", otherwise press "q + Enter" to Quit'

os.system('cls' if os.name == 'nt' else 'clear')
print(WELCOME)

# Main loop
while True:
    # Prompt the user to enter a statement
    input_statement = input(STATEMENT)

    test_news = [input_statement]
    test_news_seq = tokenizer.texts_to_sequences(test_news)
    test_news_seq = pad_sequences(test_news_seq, maxlen = maxlen, padding = padding, truncating = truncating)

    test_news_pred = model.predict(test_news_seq)

    temp = "true"
    if (np.argmax(test_news_pred[0]) == 0):
        temp = "false"
    elif(np.argmax(test_news_pred[0]) == 1):
        temp = "misleading"

    print("\n"+RESULT + temp+".\n")
    
    print(FOLLOWUP)
    # Wait for the user to press Enter to continue
    key = input()
    if key == "c":
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')
    elif  key == "q":
        sys.exit()