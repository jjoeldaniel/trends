from keybert import KeyBERT
import json
import time


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

        if count == 2000:
            break

        text = str(message['text']).strip()

        if is_invalid(text):
            continue

        messages.append(text)
        count += 1


def main(file: str):

    # Constants
    MAX_LENGTH = 3
    MIN_LENGTH = 1

    # Messages
    messages = list()

    start = time.time()

    # Read file data
    read_data(file, messages)
    message_data = '\nMessage\n'.join(str(e) for e in messages)

    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(
        message_data,
        keyphrase_ngram_range=(MIN_LENGTH, MAX_LENGTH),
        stop_words='english',
        use_mmr=True,
        diversity=0.7,
        top_n=5
    )

    ordered_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    words = {word: score for word, score in ordered_keywords}

    end = time.time()

    # Build JSON object
    json_data = {}
    info = {}

    info['current_time_epoch'] = end
    info['time_taken'] = round(end - start, 3)
    info['messages_analyzed'] = len(messages)

    json_data['info'] = info
    json_data['keywords'] = words

    # Write to file
    with open('./data/keywords.json', 'w') as f:
        f.write(json.dumps(json_data, indent=4))

    # Print status
    print('Success! Output written to ./data/keywords.json')


if __name__ == '__main__':
    main(file='./data/test_data.json')
