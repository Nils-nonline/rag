# RAG_DB
A simple and straightforward, functional rag database based on sentence_transformer. It automatically pages the RAG data to allow for RAG to work without too large pressure on the RAM. No external DB used, minimal external dependencies.

# Installation
clone this repo

``` bash
git clone https://github.com/Nils-nonline/rag.git
```

and install it's dependencies

``` bash
pip install -r requirements.txt
```

recommended python version: 3.11.7

# Getting started

## Quicksttart

have a look at the ```examples```-folder for a quickstart.

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

``` python
rag.query("hello world", k=5)
```

## Loading the DB

To load the db at the beginning of your program simply call ```.load()``` instead of ```.create()```

``` python
rag.load()
```


# FAQ

## How do I change the chunk size?
Simply change the ```.CHUNK_SIZE```-attribute of the RAG_DB object. Remember to do this each time, you load the db, with the same ```.CHUNK_SIZE``` or errors might occur.

## How do I overwrite an existing DB?
Pass force=True to the ```.create``` method

## How do I keep the DB purely in RAM?
Call ```.ram_only```, to keep the whole DB in RAM. Note that this also means that your whole DB is gone after the program ends. To keep saving the DB to a file, simply set ```.CHUNK_SIZE``` to a very large value instead of calling ```.ram_only()```
