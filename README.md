# RAG_DB
A simple and straightforward, functional rag database based on sentence_transformer. It automatically pages the RAG data to allow for RAG to work without too large pressure on the RAM. No external DB used, minimal external dependencies.

# Getting started

## Creating the DB
Create an instance of RAG_DB, passing the database-path
``` python
rag = RAG_DB("rag")
```

Now create the database using ```.create()``` (passing ```force=True``` overwrites old db at the directory, should they exist)!

``` python
rag.create()
```
## Passing data to the DB

To pass data to be queried later simply call ```.add(texts: str[])``` passing a list of strings you'd like to be added to the db.

``` python
rag.add([
  "hello world"
])
```

## Querying the DB

To query the db use the method ```query(query: str, k = 5)```, k defines how many of the top results are shown

```
rag.query("hello world", k=5)
```

## Loading the DB

To load the db at the beginning of your program simply call ```.load()``` instead of ```.create()```

```
rag.load()
```
