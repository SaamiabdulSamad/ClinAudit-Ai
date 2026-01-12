from openai import OpenAI
import os

print("ENV KEY:", os.getenv("OPENAI_API_KEY")[:10], "...")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello"}
    ]
)

print(response.choices[0].message.content)


