import rag

rag_db = rag.RAG_DB("rag")
rag_db.create()

rag_db.add(["hello_world"])
