# doctors_marketplace/services/llm.py
import os
from typing import List, Dict
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL")

class LLMClient:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # زمینهٔ ثابت برنامه (در صورت نیاز)
        messages.insert(1, {"role": "system", "content": f"آدرس پایهٔ سرویس: {BASE_URL or '—'}"})
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,  # کمی منعطف‌تر از قبل
        )
        return (resp.choices[0].message.content or "").strip()
