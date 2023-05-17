import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import re
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


def strip_text(text: str) -> str:
    """Strips text of non-essential characters.

    Returns:
        str: Cleaned text
    """

    text = text.strip()

    

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = " ".join(
        [w for w in word_tokens if not w.lower() in stop_words])

    if len(text.split()) <= 1:
        text = ""

    return lemmatize_text(text)


def group_messages(json_file_path) -> list[str]:
    """Loads and groups messages from a JSON file.

    Each index is a string of messages from a given hour
    with each message separated by a newline character.

    Returns:
        list[str]: List of messages grouped by hour
    """

    # Load the JSON file into a list of dictionaries
    with open(json_file_path, "r") as f:
        file_data = json.load(f)
        author_data = file_data["authors"]
        messages = file_data["messages"]

    authors = {}
    data = []

    # Build id : username association
    for author in author_data:
        authors[author] = author_data[author]["name"]

    # Iterate through the remaining messages
    for message in messages:
        # Get the text of the current message
        text = strip_text(message["text"])

        # Skip empty messages
        if text == "":
            continue

        message_to_be_added = (text, authors[message["author_id"]])
        data.append(message_to_be_added)

    return data


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


def build_history(messages: list[(str, str)]) -> list[(str, str)]:
    conversations = []
    current_conversation = []
    i = 0

    while i < len(messages) - 1:
        embeddings1 = model.encode(messages[i][0], convert_to_tensor=True)
        current_conversation.append(messages[i][0])

        k = i + 1
        embeddings2 = model.encode(messages[k][0], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings1, embeddings2)

        # Threshhold
        if cosine_score >= 0.55:
            current_conversation.append(messages[k])
        else:
            # Save conversation
            if len(current_conversation) > 1:
                conversations.append(current_conversation.copy())
                current_conversation.clear()

        i += 1

    return conversations


def main(file: str):
    start = time.time()

    # Messages
    messages = group_messages(file)

    history = build_history(messages)
    json_data = {}

    for i, conversation in enumerate(history):
        for message in conversation:
            # if i not in json_data:
            #     json_data[i] = [message[0]]
            # else:
            #     json_data[i] += [message[0]]
            print(message)

    # file_name = file.replace('test', 'output')
    # with open(file_name, 'w') as f:
    #     f.write(json.dumps(json_data, indent=4))

    print(f"\nTime Elapsed: {round(time.time() - start, 3)}")


if __name__ == "__main__":
    main(file="./data/test_data2.json")
