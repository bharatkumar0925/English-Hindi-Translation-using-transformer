import pandas as pd
import re
import string
import contractions

# Function to expand contractions using the contractions library
def expand_contractions(text):
    return contractions.fix(text)

# Function to remove punctuation but keep digits
def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)

# Function to remove unnecessary spaces
def clean_spaces(text):
    return re.sub(" +", " ", text).strip()

# Main cleaning function
def clean_text(data, source_col, target_col):
    # Lowercase
    data[source_col] = data[source_col].str.lower()
    data[target_col] = data[target_col].str.lower()

    # Expand contractions
    data[source_col] = data[source_col].apply(expand_contractions)

    # Remove punctuation (digits are preserved)
    data[source_col] = data[source_col].apply(remove_punctuation)
    data[target_col] = data[target_col].apply(remove_punctuation)

    # Remove specific Hindi symbols (like danda "।")
    hindi_remove_symbols = "।"
    data[target_col] = data[target_col].apply(lambda x: ''.join(c for c in x if c not in hindi_remove_symbols))

    # Remove extra spaces
    data[source_col] = data[source_col].apply(clean_spaces)
    data[target_col] = data[target_col].apply(clean_spaces)

    # Add start and end tokens to target
    data[target_col] = data[target_col].apply(lambda x: 'START_ ' + x + ' _END')

    # Extract vocab
    source_words = set(word for sentence in data[source_col] for word in sentence.split())
    target_words = set(word for sentence in data[target_col] for word in sentence.split())

    print("Source vocab size:", len(source_words))
    print("Target vocab size:", len(target_words))
    return source_words, target_words
