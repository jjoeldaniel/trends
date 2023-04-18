import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
import time


# Download NLTK resources (if not already downloaded)
nltk.download('wordnet')

# Load BERTopic model
topic_model = BERTopic()


def is_invalid(message: str):
    '''Validates message

    Returns:
        bool: True if message is invalid, False otherwise
    '''

    # Messages that are empty
    if message == '':
        return True
    # Messages that are just links
    elif message.startswith('http') and len(message.split(' ')) == 1:
        return True
    # Messages that are just emojis
    elif (message.startswith(':') and
          message.endswith(':') and
          len(message.split(' ')) == 1):
        return True

    else:
        return False


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []
    for word, pos in nltk.pos_tag(nltk.word_tokenize(text)):
        pos = pos[0].lower()
        pos = pos if pos in ['a', 's', 'r', 'n', 'v'] else None
        lemmatized_text.append(lemmatizer.lemmatize(word, pos=pos) if pos else word)
    return " ".join(lemmatized_text)


def read_data(file: str, messages: list):
    '''Reads data from JSON file

    Returns:
        messages: list of messages
    '''

    # Load data
    with open(file) as f:
        data = json.load(f)

    count = 0
    for message in data['messages']:

        if count == 3000:
            break

        text = str(message['text']).strip()

        if is_invalid(text):
            continue

        # lemmatize!!
        text = lemmatize_text(text)

        messages.append(text)
        count += 1


def main(file: str):

    # Messages
    messages = list()

    start = time.time()

    # Read file data
    read_data(file, messages)
    message_data = ''.join(str(e) for e in messages)
    topics, _ = topic_model.transform([message_data])

    # Get topic labels and probabilities
    topic_labels = topics[0]
    topic_probabilities = topics[1]

    # Print topic labels and probabilities
    print("Topic Labels: ", topic_labels)
    print("Topic Probabilities: ", topic_probabilities)

    end = time.time()
    print(round(end - start, 3))


if __name__ == '__main__':
    main(file='./data/test_data.json')
