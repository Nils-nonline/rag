import rag

# after having called .create in aprevious session and added texts to it you can now load the db again with .load
rag_db = rag.RAG_DB("rag")
rag_db.load()

# search for text simlar to 'hello world' and display the top 5 results
rag_db.query("hello world", k = 5)
