"""Twitter/X Thread Generator — Converts long-form content into engaging tech threads.

Handles:
- Thread splitting (280 char limit per tweet, numbered 1/N format)
- Context-aware hashtag injection
- Developer-first tone enforcement
- Platform-specific formatting
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from app.logger import get_logger

logger = get_logger("twitter_thread")

# ═══════════════════════════════════════════════════════════════════════════════
#  Hashtag Engine
# ═══════════════════════════════════════════════════════════════════════════════

# Keyword → hashtag mapping for tech content
HASHTAG_RULES: dict[str, list[str]] = {
    "rust": ["#RustLang"],
    "python": ["#Python"],
    "typescript": ["#TypeScript"],
    "javascript": ["#JavaScript"],
    "golang": ["#Golang"],
    "open-source": ["#OpenSource"],
    "open source": ["#OpenSource"],
    "docker": ["#Docker"],
    "kubernetes": ["#Kubernetes"],
    "llm": ["#LLM", "#AI"],
    "ai": ["#AI"],
    "machine learning": ["#MachineLearning"],
    "search engine": ["#SearchEngine"],
    "meta-search": ["#MetaSearch"],
    "web scraping": ["#WebScraping"],
    "api": ["#API"],
    "async": ["#Async"],
    "tokio": ["#RustLang", "#Tokio"],
    "privacy": ["#Privacy"],
    "self-host": ["#SelfHosted"],
    "github": ["#GitHub"],
    "devtools": ["#DevTools"],
}

# Maximum hashtags per thread to avoid looking spammy
MAX_HASHTAGS = 4
# Maximum tweet length
MAX_TWEET_LENGTH = 280


def _extract_hashtags(content: str, max_tags: int = MAX_HASHTAGS) -> list[str]:
    """Extract relevant hashtags from content based on keyword matching."""
    lower = content.lower()
    found_tags: list[str] = []
    seen: set[str] = set()

    for keyword, tags in HASHTAG_RULES.items():
        if keyword in lower:
            for tag in tags:
                if tag.lower() not in seen and len(found_tags) < max_tags:
                    found_tags.append(tag)
                    seen.add(tag.lower())

    return found_tags


# ═══════════════════════════════════════════════════════════════════════════════
#  Thread Generator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Tweet:
    """A single tweet in a thread."""
    content: str
    index: int
    total: int

    @property
    def formatted(self) -> str:
        """Return the tweet with thread numbering."""
        if self.total == 1:
            return self.content
        return f"{self.content}\n\n({self.index}/{self.total})"

    @property
    def length(self) -> int:
        return len(self.formatted)


@dataclass
class TwitterThread:
    """A complete thread of tweets."""
    tweets: list[Tweet] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)

    @property
    def total_tweets(self) -> int:
        return len(self.tweets)


class ThreadGenerator:
    """Converts long-form content into a Twitter/X thread.
    
    Usage:
        gen = ThreadGenerator()
        thread = gen.generate(title, body, github_url)
        for tweet in thread.tweets:
            print(tweet.formatted)
    """

    def __init__(self, max_tweet_length: int = MAX_TWEET_LENGTH):
        self.max_length = max_tweet_length

    def generate(
        self,
        title: str,
        body: str,
        github_url: str = "",
        extra_hashtags: list[str] | None = None,
    ) -> TwitterThread:
        """Generate a thread from long-form content.
        
        Strategy:
        - Tweet 1: Hook (title + value prop)
        - Tweets 2-N: Key points, one per tweet
        - Last tweet: CTA with GitHub link + hashtags
        """
        # Extract hashtags from the full content
        full_content = f"{title} {body}"
        hashtags = _extract_hashtags(full_content)
        if extra_hashtags:
            for tag in extra_hashtags:
                if tag not in hashtags and len(hashtags) < MAX_HASHTAGS:
                    hashtags.append(tag)

        # Build the raw tweet segments
        segments = self._split_into_segments(title, body, github_url, hashtags)

        # Create numbered tweets
        total = len(segments)
        tweets = [
            Tweet(content=seg, index=i + 1, total=total)
            for i, seg in enumerate(segments)
        ]

        # Validate lengths and split any that are too long
        tweets = self._enforce_length_limits(tweets)

        thread = TwitterThread(tweets=tweets, hashtags=hashtags)
        logger.info(
            "Generated %d-tweet thread with hashtags: %s",
            thread.total_tweets, ", ".join(hashtags),
        )
        return thread

    def _split_into_segments(
        self,
        title: str,
        body: str,
        github_url: str,
        hashtags: list[str],
    ) -> list[str]:
        """Split content into logical tweet-sized segments."""
        segments: list[str] = []

        # Tweet 1: Hook
        hook = f"🧵 {title}"
        if len(hook) > self.max_length - 10:
            hook = hook[: self.max_length - 15] + "..."
        segments.append(hook)

        # Middle tweets: Parse body into logical chunks
        # Split on double newlines (paragraphs) or bullet points
        paragraphs = re.split(r"\n\n+", body.strip())

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If it's a bullet list, group bullets together
            if para.startswith("- ") or para.startswith("• "):
                lines = para.split("\n")
                current_chunk = ""
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    test = f"{current_chunk}\n{line}" if current_chunk else line
                    if len(test) < self.max_length - 10:
                        current_chunk = test
                    else:
                        if current_chunk:
                            segments.append(current_chunk)
                        current_chunk = line
                if current_chunk:
                    segments.append(current_chunk)
            else:
                # Regular paragraph
                if len(para) < self.max_length - 10:
                    segments.append(para)
                else:
                    # Split long paragraphs on sentence boundaries
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        test = f"{current_chunk} {sent}" if current_chunk else sent
                        if len(test) < self.max_length - 10:
                            current_chunk = test
                        else:
                            if current_chunk:
                                segments.append(current_chunk)
                            current_chunk = sent
                    if current_chunk:
                        segments.append(current_chunk)

        # Final tweet: CTA with GitHub link + hashtags
        hashtag_str = " ".join(hashtags)
        if github_url:
            cta = f"Check it out 👇\n{github_url}\n\n{hashtag_str}"
        else:
            cta = f"Would love to hear your thoughts!\n\n{hashtag_str}"

        segments.append(cta.strip())
        return segments

    def _enforce_length_limits(self, tweets: list[Tweet]) -> list[Tweet]:
        """Ensure no tweet exceeds the character limit."""
        valid: list[Tweet] = []
        for tweet in tweets:
            if tweet.length <= self.max_length:
                valid.append(tweet)
            else:
                # Force-split at the character limit
                content = tweet.content
                while content:
                    chunk = content[: self.max_length - 15]
                    content = content[self.max_length - 15:]
                    valid.append(Tweet(content=chunk, index=0, total=0))

        # Re-number after potential splits
        total = len(valid)
        for i, t in enumerate(valid):
            t.index = i + 1
            t.total = total

        return valid
