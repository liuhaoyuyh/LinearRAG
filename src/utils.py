from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os
from dotenv import load_dotenv
from src.model_client import ModelRequest, ModelResponse, create_model_client_from_env
load_dotenv()
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    def __init__(self, llm_model):
        self._client = create_model_client_from_env(default_model=llm_model)
        self._max_tokens = 2000
        self._temperature = 0

    def generate(self, messages) -> ModelResponse:
        return self._client.generate(
            ModelRequest(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        )

    def infer(self, messages):
        return self.generate(messages).content


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    s = str(text)
    cjk = re.findall(r"[\u4e00-\u9fff]", s)
    cjk_count = len(cjk)
    ascii_words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", s)
    ascii_word_count = len(ascii_words)
    non_space = len(re.findall(r"\S", s))
    other = max(0, non_space - cjk_count - sum(len(w) for w in ascii_words))
    est = cjk_count + ascii_word_count + int(other / 4)
    return max(1, est) if est > 0 else 0


def estimate_message_tokens(messages) -> int:
    if not messages:
        return 0
    total = 0
    for msg in messages:
        try:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            total += estimate_token_count(role) + estimate_token_count(content)
        except Exception:
            continue
    return total



def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
