import json
import google.generativeai as palm
from dotenv import load_dotenv
import os

load_dotenv()
palm.configure(api_key=os.getenv("API_KEY"))


def read_data() -> str:
    with open("./data/filtered_data.txt", "r") as f:
        lines = f.readlines()
    return "\n".join(lines)


def is_valid_response(response: palm.types.Completion) -> bool:
    if response.result is None:
        filters = response.filters
        safety_feedback = response.safety_feedback

        if filters is not None and len(filters) > 0:
            print(filters)
        if safety_feedback is not None and len(safety_feedback) > 0:
            print(safety_feedback)
        return False
    return True


def get_trends(messages: str) -> list:
    defaults = {
        "model": "models/text-bison-001",
        "temperature": 1.0,
        "candidate_count": 1,
        "top_k": 100,
        "top_p": 0.95,
        "max_output_tokens": 3072,
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

    # Check for errors
    if not is_valid_response(response):
        raise Exception("Failed to generate text")

    result: str = response.result.replace(", ", ",")

    return result.split(",")


def main():
    messages = read_data()
    trends = {"trends": get_trends(messages)}

    with open("./data/trends.json", "w") as f:
        json.dump(trends, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
