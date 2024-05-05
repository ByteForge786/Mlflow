from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pretrained sentence transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def is_related_to_schema(query):
    # Encode column names and user query into embeddings
    column_embeddings = model.encode(column_names)
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity between query embedding and each column name embedding
    similarities = cosine_similarity([query_embedding], column_embeddings)[0]
    
    # Set a threshold for similarity score
    threshold = 0.5
    
    # Return True if maximum similarity score is above the threshold
    return max(similarities) > threshold

# Sample conversation loop
while True:
    user_input = input("User: ")
    
    if is_related_to_schema(user_input):
        print("Chatbot: The query is related to the schema.")
        # Process the query with respect to the schema
    else:
        print("Chatbot: The query is not related to the schema.")
        # Optionally, prompt the user to ask a query related to the schema




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample list of column names
column_names = ["name", "age", "address", "email", "phone"]

def is_related_to_schema(query):
    # Convert column names to a single string
    column_text = ', '.join(column_names)
    
    # Vectorize column text and user query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([column_text, query])
    
    # Calculate cosine similarity between column text and user query
    similarity = cosine_similarity(vectors)[0][1]
    
    # Set a threshold for similarity score
    threshold = 0.5
    
    # Return True if similarity score is above the threshold
    return similarity > threshold

# Sample conversation loop
while True:
    user_input = input("User: ")
    
    if is_related_to_schema(user_input):
        print("Chatbot: The query is related to the schema.")
        # Process the query with respect to the schema
    else:
        print("Chatbot: The query is not related to the schema.")
        # Optionally, prompt the user to ask a query related to the schema




 import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pretrained sentence transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def split_into_sentences(query, keywords):
    # Split the query into sentences based on the keywords
    sentences = re.split(r'\b(?:{})\b'.format('|'.join(keywords)), query, flags=re.IGNORECASE)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def is_related_to_schema(sentence, column_embeddings, threshold=0.5):
    # Encode the sentence into embeddings
    sentence_embedding = model.encode(sentence)
    
    # Calculate cosine similarity between the sentence embedding and each column name embedding
    similarities = cosine_similarity([sentence_embedding], column_embeddings)
    
    # Return True if any similarity score is above the threshold
    return any(similarity > threshold for similarity in similarities.flatten())

# Sample list of column names
column_names = ["rule_id", "name", "description"]

# Sample conversation loop
while True:
    user_input = input("User: ")
    
    # Define the keywords used to split the input into sentences
    keywords = ["and", "or", "but", "however", "although", "because", "since", "when"]
    
    # Split the user input into sentences
    sentences = split_into_sentences(user_input, keywords)
    
    # Encode column names into embeddings
    column_embeddings = model.encode(column_names)
    
    # Initialize a flag to indicate if any sentence is related to the schema
    related_to_schema = False
    
    # Iterate through each sentence
    for sentence in sentences:
        # Check if the sentence is related to the schema
        if is_related_to_schema(sentence, column_embeddings):
            related_to_schema = True
            break
    
    if related_to_schema:
        print("Chatbot: The query is related to the schema.")
        # Process the query with respect to the schema
        
        # Update context flag to True if necessary
    else:
        print("Chatbot: The query is not related to the schema.")
        # Optionally, prompt the user to ask a query related to the schema
        
        # Update context flag to False if necessary






