import json
import nltk
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
import time
import emoji
import dateutil.parser as dp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# Load BERTopic model
topic_model = BERTopic()


def strip_text(text: str) -> str:
    """Strips text of non-essential characters.

    Performs the following operations:
        - Remove emojis
        - Remove URLs
        - Remove HTML tags
        - Remove punctuation

    Returns:
        str: Cleaned text
    """

    text = emoji.demojize(text).strip()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<\S+", "", text)
    text = re.sub(r":\S+", "", text)
    text = re.sub(r"__\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    text = lemmatize_text(text)

    # Remove stopwords
    text_tokens = word_tokenize(text)
    tokens_without_sw = [
        word for word in text_tokens if word not in stopwords.words()]
    text = " ".join(tokens_without_sw)

    return text.strip()


def group_messages(json_file_path) -> list[str]:
    """Loads and groups messages from a JSON file.

    Each index is a string of messages from a given hour
    with each message separated by a newline character.

    Returns:
        list[str]: List of messages grouped by hour
    """

    # Load the JSON file into a list of dictionaries
    with open(json_file_path, "r") as f:
        messages = json.load(f)["messages"]

    # Get the timestamp of the first message in epoch time
    current_time = dp.parse(messages[0]["timestamp"]).timestamp()
    current_messages = []

    # Initialize the array to store the grouped messages
    grouped_messages = []

    # Iterate through the remaining messages
    for message in messages:
        # Get the text of the current message
        text = strip_text(message["text"])

        # Skip empty messages
        if text == "":
            continue

        # Get the timestamp of the current message as a datetime object
        timestamp = dp.parse(message["timestamp"]).timestamp()

        # If the current message is within an hour of the previous message,
        # add it to the current group
        if abs(timestamp - current_time) < 3600:
            current_messages.append(text)
        # Otherwise, add the current group to the grouped messages array
        # and start a new group
        else:
            grouped_messages.append("\n".join(current_messages))
            current_messages = [text]

        # Update the current time
        current_time = timestamp

    # Add the last group to the grouped messages array
    grouped_messages.append("\n".join(current_messages))

    return grouped_messages


def lemmatize_text(text) -> str:
    """Lemmatizes text

    Returns:
        str: Lemmatized text
    """

    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []

    for word, pos in nltk.pos_tag(nltk.word_tokenize(text)):
        pos = pos[0].lower()
        pos = pos if pos in ["a", "s", "r", "n", "v"] else None
        lemmatized_text.append(lemmatizer.lemmatize(
            word, pos=pos) if pos else word)

    return " ".join(lemmatized_text)


def main(file: str):
    start = time.time()

    # Messages
    messages = group_messages(file)

    # Fit the topic model
    # topics, _ = topic_model.fit_transform(messages)

    end = time.time()
    print(f"\nTime Elapsed: {round(end - start, 3)}\n")

    for m in messages:
        print(m + "\n")


if __name__ == "__main__":
    main(file="./data/test_data2.json")
