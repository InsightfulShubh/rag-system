import os
import time
import random
import re
import requests
from dotenv import load_dotenv

# ==============================
# CONFIG
# ==============================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "llama-3.1-8b-instant"   # stable + low TPM
TEMPERATURE = 0.6
MAX_TOKENS = 1200               # reduced to avoid TPM limit
RATE_LIMIT_SECONDS = 5          # base delay
MAX_RETRIES = 4

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPICS = [
    # "Artificial Intelligence",
    # "Machine Learning",
    # "Deep Learning",
    # "Cloud Computing",
    # "AWS Services",
    # "Diabetes",
    # "Heart Disease",
    # "Python Programming",
    # "FastAPI Framework",
    # "Docker Basics",
    # "Kubernetes Architecture",
    # "Microservices Design",
    # "API Gateway",
    # "Cyber Security",
    # "Data Privacy",
    # "DevOps Practices",
    "CI_CD Pipelines",
    "Database Systems",
    "NoSQL Databases",
    "Distributed Systems"
]

# ==============================
# PROMPT TEMPLATE
# ==============================

def build_prompt(topic: str) -> str:
    return f"""
Generate a detailed and well-structured document (800–1200 words) on the topic: "{topic}"

Requirements:
- Start with a clear definition
- Explain key concepts in detail
- Include real-world applications
- Mention advantages and challenges
- Provide practical examples

Important:
- Use clear paragraphs (no bullet-only content)
- Maintain a professional tone
- Ensure readability and coherence
- Add slight overlap with related topics (AI, cloud, distributed systems) where relevant
- Avoid repetition

Output must be clean plain text (no markdown).
"""

# ==============================
# HELPER: Extract retry time
# ==============================

def extract_retry_time(error_text: str):
    match = re.search(r"try again in ([\d\.]+)s", error_text)
    if match:
        return float(match.group(1))
    return None

# ==============================
# GROQ API CALL
# ==============================

def generate_content(topic: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert technical writer producing high-quality knowledge base content."},
            {"role": "user", "content": build_prompt(topic)}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]

            else:
                error_text = response.text
                print(f"[WARN] Attempt {attempt} failed for {topic}: {error_text}")

                retry_time = extract_retry_time(error_text)

                if retry_time:
                    sleep_time = retry_time + random.uniform(1, 3)
                else:
                    sleep_time = (2 ** attempt) + random.uniform(1, 2)

                print(f"[INFO] Sleeping for {sleep_time:.2f}s before retry...")
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[ERROR] Attempt {attempt} exception for {topic}: {e}")

            sleep_time = (2 ** attempt) + random.uniform(1, 2)
            print(f"[INFO] Sleeping for {sleep_time:.2f}s after exception...")
            time.sleep(sleep_time)

    return None

# ==============================
# SAVE FILE
# ==============================

import re

def sanitize_filename(name: str) -> str:
    """
    Remove invalid characters for file names
    """
    name = name.replace(" ", "_")
    name = re.sub(r'[<>:"/\\|?*]', '', name)  # remove invalid chars
    return name


def save_file(topic: str, content: str):
    safe_name = sanitize_filename(topic)
    filename = safe_name + ".txt"

    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ==============================
# MAIN
# ==============================

def main():
    print("Starting KB generation...\n")

    for topic in TOPICS:
        print(f"[INFO] Generating: {topic}")

        content = generate_content(topic)

        if content:
            save_file(topic, content)
            print(f"[SUCCESS] Saved: {topic}")
        else:
            print(f"[FAILED] Could not generate: {topic}")

        # Base delay between topics
        sleep_time = RATE_LIMIT_SECONDS + random.uniform(1, 2)
        print(f"[INFO] Waiting {sleep_time:.2f}s before next topic...\n")
        time.sleep(sleep_time)

    print("\nKB generation completed.")

# ==============================
# ENTRY
# ==============================

if __name__ == "__main__":
    main()