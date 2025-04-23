import re


def clean_text(text: str) -> str:
    """
    Clean the text by removing HTML tags and extra spaces.
    :param text: the text to clean
    :return: cleaned text
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
