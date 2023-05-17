import json
import nltk
import nltk.data
from nltk.stem.snowball import SnowballStemmer
import time
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Download NLTK resources (if not already downloaded)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


def strip_text(text: str) -> str:
    """Strips text of non-essential characters.

    Returns:
        str: Cleaned text
    """

    sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    text = "\n-----\n".join(sent_detector.tokenize(text.strip()))

    return text.strip()


def group_messages(json_file_path) -> list[(str, str)]:
    """Loads and groups messages from a JSON file.

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

        # Skip empty and one-worded messages
        if text == "" or len(text.split(" ")) == 1:
            continue

        message_to_be_added = (authors[message["author_id"]], text)
        data.append(message_to_be_added)

    return data


def stemm_text(text) -> str:
    """Stemms text using NLTK's Snowball stemmer.

    Stemmers remove morphological affixes from words, leaving
    only the word stem.

    Keyword arguments:
    `text` -- text to be stemmed
    Return: stemmed text
    """

    return SnowballStemmer("english").stem(text)


def build_conversations(messages: list[(str, str)]) -> list[list[(str, str)]]:
    """Builds a list of conversations

    Keyword arguments:
    `messages` -- list of messages in the form (author, message) and sorted by
    time

    Return: list of conversations grouped by cosine similarity
    """

    conversations = []
    current_conversation = []
    i = 0

    while i < len(messages):
        embeddings1 = model.encode(messages[i][1], convert_to_tensor=True)

        if (messages[i][0], messages[i][1]) not in current_conversation:
            current_conversation.append((messages[i][0], messages[i][1]))

        k = i + 1

        # Check if k is out of bounds
        if k >= len(messages):
            # Only add to conversations if there are 2 or more messages
            if len(current_conversation) > 1:
                conversations.append(current_conversation.copy())

            break

        embeddings2 = model.encode(messages[k][1], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings1, embeddings2)

        # Threshhold
        if cosine_score >= 0.2:
            current_conversation.append((messages[k][0], messages[k][1]))
        else:
            # Only add to conversations if there are 2 or more messages
            if len(current_conversation) > 1:
                conversations.append(current_conversation.copy())

            current_conversation.clear()

        i += 1

    return conversations


def main(file: str):
    start = time.time()

    # Output data
    json_data = {}

    # List of pairs (author, message)
    messages = group_messages(file)

    # List of conversations (which are list[(str, str)])
    history = build_conversations(messages)

    # Iterate through all conversations
    for i, conversation in enumerate(history):
        # Each index of json_data is an array of messages with a
        # author and text field

        author_messages = []
        for messages in conversation:
            x = {}
            x["author"] = messages[0]
            x["message"] = messages[1]
            author_messages.append(x)

        json_data[i] = author_messages

    # Write to output file
    file_name = file.replace("test", "output")
    with open(file_name, "w") as f:
        f.write(json.dumps(json_data, indent=4))

    print(f"\nTime Elapsed: {round(time.time() - start, 3)}")


if __name__ == "__main__":
    main(file="./data/test_data2.json")
