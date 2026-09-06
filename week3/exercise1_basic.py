import requests

def ask_codellama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "codellama:7b",
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    question = "Write a Python function to check if a number is a palindrome."
    answer = ask_codellama(question)
    print("Question:", question)
    print("\nAnswer:\n", answer)
