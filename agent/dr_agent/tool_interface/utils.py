"""
Utility functions for MCP agents, including string processing and snippet localization.
"""

import re
from typing import List, Tuple

# Import for sentence tokenization with fallback
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    # Fallback if nltk is not available
    def sent_tokenize(text: str) -> List[str]:
        return re.split(r"(?<=[.!?]) +", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text for better matching"""
    return re.sub(r"[^\w\s]", " ", text)


def f1_score(set1: set, set2: set) -> float:
    """Calculate F1 score between two sets of words"""
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    if intersection == 0:
        return 0.0

    precision = intersection / len(set1)
    recall = intersection / len(set2)

    return 2 * precision * recall / (precision + recall)


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 3000
) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful,
                         the second element is the extracted context.
    """
    try:
        # Limit full text to prevent excessive processing
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2  # Minimum threshold

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first part of the full text
            return False, full_text[: context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"
