import google.generativeai as palm
from dotenv import load_dotenv
import os

load_dotenv()
palm.configure(api_key=os.getenv("API_KEY"))


def read_data() -> str:
    with open("./data/filtered_data.txt", "r") as f:
        lines = f.readlines()
    return "\n".join(lines)


def get_trends(messages: str) -> list:
    defaults = {
        "model": "models/text-bison-001",
        "temperature": 0.7,
        "candidate_count": 1,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "stop_sequences": [],
        "safety_settings": [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 3},
            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 3},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 3},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 3},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 3},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 3},
        ],
    }

    prompt = f"""
    Given this list of Discord messages,
    return a list of topics discussed. Avoid duplicates and stopwords

    {messages}

    Topics: """

    response = palm.generate_text(**defaults, prompt=prompt)
    print(response.result)


def main():
    messages = read_data()
    get_trends(messages)


if __name__ == "__main__":
    main()
