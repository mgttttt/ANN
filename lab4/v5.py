import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    data['cleaned_text'] = data['tweet_text'].apply(lambda x: re.sub(r'http\S+', '', x))
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub(r'@\S+', '', x))
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub(r'#\S+', '', x))
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x.lower()))

    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['cyberbullying_type'])

    return data, label_encoder.classes_

def vectorize_text(data, max_words=1000):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
    max_length = max(len(x) for x in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    return padded_sequences, max_length

def create_model(vocabulary_size, max_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim=50, input_length=max_length))
    model.add(GRU(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(6, activation='softmax'))  # Предполагаем 6 классов
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    file_path = 'cyberbullying_tweets.csv'
    data, classes = load_and_preprocess_data(file_path)
    X, max_length = vectorize_text(data, max_words=1000)
    y = to_categorical(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(1000, max_length)

    model.fit(X_train, y_train, epochs=1, batch_size=64, validation_split=0.2)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Точность модели: {accuracy}')
