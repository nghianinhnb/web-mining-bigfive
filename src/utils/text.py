import re
from typing import List


def preprocess_tweet(text: str) -> str:
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    text = " ".join(text.split())
    return text.strip()


def preprocess_tweets(tweets: List[str]) -> List[str]:
    return [preprocess_tweet(t) for t in tweets if t.strip()]


def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#(\w+)", text.lower())


def extract_all_hashtags(tweets: List[str]) -> List[str]:
    hashtags = []
    for tweet in tweets:
        hashtags.extend(extract_hashtags(tweet))
    return hashtags


def truncate_text(text: str, max_tokens: int = 64) -> str:
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text

