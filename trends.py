from keybert import KeyBERT
import json
import time

def is_invalid(message: str):
    
    # Messages that are empty
    if message == '':
        return True
        
    # Messages that are just links
    elif message.startswith('http') and len(message.split(' ')) == 1:
        return True
    
    # Messages that are just emojis
    elif message.startswith(':') and message.endswith(':') and len(message.split(' ')) == 1:
        return True
    
    else:
        return False

def main(file='test_data.json'):
    
    # Constants
    MAX_LENGTH = 3
    MIN_LENGTH = 1
    
    # Messages
    messages = list()
    
    start = time.time()
    
    # Load data
    with open(file) as f:
        data = json.load(f)

    for message in data['messages']:
        
        text = str(message['text']).strip()
        
        if is_invalid(text):
            continue
        
        messages.append(text)

    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords('\nMessage\n'.join(str(e) for e in messages), keyphrase_ngram_range=(MIN_LENGTH, MAX_LENGTH), stop_words='english', use_mmr=True, diversity=0.7, top_n=5)
    words = [x[0] for x in keywords]

    end = time.time()

    # Print results
    print(f'Time taken: {round(end - start, 2)} seconds')
    print(f'Messages analyzed: {len(messages)}')
    print(f'Keywords: {words}')

if __name__ == '__main__':
    main(file='./data/test_data.json')