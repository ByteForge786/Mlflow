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

def extract_conditions(query):
    # Define keywords and syntax patterns to identify conditions
    keywords = ["where", "and", "or"]
    syntax_patterns = ["[a-zA-Z_]+\s*=\s*[^\s,]+", "[a-zA-Z_]+\s*like\s*'%[^']*%'"]
    
    # Initialize list to store extracted conditions
    conditions = []
    
    # Iterate through keywords and syntax patterns to extract conditions
    for keyword in keywords:
        idx = query.lower().find(keyword)
        if idx != -1:
            for pattern in syntax_patterns:
                matches = re.findall(pattern, query[idx:], flags=re.IGNORECASE)
                conditions.extend(matches)
    
    return conditions

def is_related_to_schema(query):
    # Define keywords related to SQL queries
    keywords = ["select", "from", "where", "and", "or", "not", "between", "like", "in", "is", 
                "null", "order by", "group by", "having", "limit", "offset", "distinct", 
                "join", "left join", "right join", "inner join", "outer join", "union", 
                "intersect", "except"]
    
    # Split the query into sentences based on the presence of keywords
    sentences = re.split(r'\b(?:{})\b'.format('|'.join(keywords)), query, flags=re.IGNORECASE)
    
    # Encode column names into embeddings
    column_embeddings = model.encode(column_names)
    
    # Initialize a flag to indicate if the query is related to the schema
    related_to_schema = True
    
    # Iterate through each sentence
    for sentence in sentences:
        # Extract conditions from the sentence
        conditions = extract_conditions(sentence)
        
        if conditions:
            # Encode conditions into embeddings
            condition_embeddings = model.encode(conditions)
            
            # Calculate cosine similarity between each condition embedding and each column name embedding
            similarities = cosine_similarity(condition_embeddings, column_embeddings)
            
            # Check if any similarity score is below the threshold
            if any(similarity < 0.5 for similarity in similarities.flatten()):
                related_to_schema = False
                break
    
    return related_to_schema

# Sample list of column names
column_names = ["rule_id", "name", "description"]

# Sample conversation loop
while True:
    user_input = input("User: ")
    
    if is_related_to_schema(user_input):
        print("Chatbot: The query is related to the schema.")
        # Process the query with respect to the schema
        
        # Update context flag to True if necessary
    else:
        print("Chatbot: The query is not related to the schema.")
        # Optionally, prompt the user to ask a query related to the schema
        
        # Update context flag to False if necessary





