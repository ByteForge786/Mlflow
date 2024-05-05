 def split_sentence_with_keywords(sentence, keywords):
    # Tokenize the sentence into words and punctuation marks
    tokens = re.findall(r'\w+|[^\w\s]', sentence)
    
    # Initialize a list to store the split parts of the sentence
    split_parts = []
    current_part = []
    
    # Iterate through the tokens in the sentence
    for token in tokens:
        # Check if the token is a keyword
        if token.lower() in keywords:
            # If it is a keyword, add the current part to the split parts list
            if current_part:
                split_parts.append(' '.join(current_part))
                current_part = []
        else:
            # If it is not a keyword, add it to the current part
            current_part.append(token)
    
    # Add the last part to the split parts list
    if current_part:
        split_parts.append(' '.join(current_part))
    
    return split_parts

# Example usage:
import re

sentence = "Today, I eat potato. I am happy."
keywords = ["and", "or", "not", "."]

split_parts = split_sentence_with_keywords(sentence, keywords)
print("Split parts of the sentence:", split_parts)
