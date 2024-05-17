response = [{'generated_text': '###Instruction: some random text. ###Response: '''sql\nselect *\n from table where id=2344\n'''}]

# Extract the SQL query from the response
sql_query = response[0]['generated_text'].split("'''sql")[1].split("'''")[0].strip()

# Remove newline characters from the SQL query
sql_query_cleaned = sql_query.replace('\n', '')

print(sql_query_cleaned)
