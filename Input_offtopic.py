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




from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pretrained sentence transformer model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def is_related_to_schema(user_input, column_names, threshold=0.5):
    # Convert column names to lowercase for case-insensitive comparison
    column_names_lower = set(name.lower() for name in column_names)
    
    # Define keywords for splitting user input into sentences
    keywords = ["and", "or", "but", "however", "although", "because", "since", "when", ".", ",", ";", ":", "!"]
    
    # Split the user input into sentences based on the keywords
    sentences = [user_input]
    for keyword in keywords:
        split_sentences = []
        for sentence in sentences:
            split_sentences.extend(sentence.split(keyword))
        sentences = split_sentences
    
    # Iterate through each sentence
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
        
        # Convert the sentence to lowercase
        sentence_lower = sentence.lower()
        
        # Encode column names into embeddings
        column_embeddings = model.encode(column_names_lower)
        
        # Encode the sentence into embeddings
        sentence_embedding = model.encode([sentence_lower])
        
        # Calculate cosine similarity between sentence embedding and column name embeddings
        similarities = cosine_similarity(sentence_embedding, column_embeddings)
        
        # Check if any similarity score is below the threshold
        if any(similarity < threshold for similarity in similarities.flatten()):
            return False
    
    return True

# Sample list of column names
column_names = ["rule_id", "name", "description"]

# Sample conversation loop
while True:
    user_input = input("User: ")
    
    # Check if the user input is related to the schema
    related_to_schema = is_related_to_schema(user_input, column_names)
    
    if related_to_schema:
        print("Chatbot: The query is related to the schema.")
        # Process the query with respect to the schema
        
        # Update context flag to True if necessary
    else:
        print("Chatbot: The query is not related to the schema.")
        # Optionally, prompt the user to ask a query related to the schema
        
        # Update context flag to False if necessary
