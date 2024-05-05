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
