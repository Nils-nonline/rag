import math
import os
import shutil

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def normalize(file_name):
    if file_name[-1] != "/":
        return file_name + "/"
    return file_name


class RAG_DB:
    def __init__(self, file_name):
        self.CHUNK_SIZE = 20
        self.SEPERATOR = "\n@SEP@\n"
        self.RAM_ONLY = False

        self.file_name = normalize(file_name)
        self.cursor = None
        self.db_end = None
        self.data = None

    def ram_only(self):
        self.RAM_ONLY = True
        self.CHUNK_SIZE = 1_000_000_000
    def create(self, force=False):
        if not self.RAM_ONLY:
            if not os.path.exists(self.file_name):
                os.makedirs(self.file_name)
            else:
                if not force:
                    raise Exception("db already exists")
                else:
                    shutil.rmtree(self.file_name)
                    os.mkdir(self.file_name)
    
            with open(f"{self.file_name}data.json", "w") as f:
                f.write("0")

        self.cursor = 0
        self.db_end = 0
        self.embeddings = None
        self.texts = None

    def load(self):
        self.file_name = normalize(self.file_name)

        if not os.path.exists(self.file_name):
            raise Exception("db does not exist")

        with open(f"{self.file_name}data.json", "r") as f:
            self.db_end = int(f.read())

        self.embeddings = None
        self.texts = None

    def save_embeddings(self, chunk):
        np.save(f"{self.file_name}embed-{chunk}.npy", self.embeddings)

    def save_texts(self, chunk):
        with open(f"{self.file_name}text-{chunk}.txt", "w") as file:
            file.write(self.SEPERATOR.join(self.texts))

    def pos2chunk(self, pos):
        return math.ceil(pos / self.CHUNK_SIZE)

    def save_chunk(self, chunk, to_save="both"):
        if self.RAM_ONLY: return
        
        if to_save == "embedding":
            self.save_embeddings(chunk)

        if to_save == "texts":
            self.save_texts(chunk)

        if to_save == "both":
            self.save_embeddings(chunk)
            self.save_texts(chunk)

    def save_position(self, position, to_save="both"):
        self.save_chunk(self.pos2chunk(position), to_save)

    def load_embeddings(self, chunk):
        if not os.path.exists(f"{self.file_name}embed-{chunk}.npy"):
            self.embeddings = None
            return

        self.embeddings = np.load(f"{self.file_name}embed-{chunk}.npy")

    def load_texts(self, chunk):
        if not os.path.exists(f"{self.file_name}text-{chunk}.txt"):
            self.texts = None
            return

        with open(f"{self.file_name}text-{chunk}.txt", "r") as file:
            self.texts = file.read().split(self.SEPERATOR)

    def load_position(self, position, to_load="both"):
        self.load_chunk(self.pos2chunk(position), to_load)

    def load_chunk(self, chunk, to_load="both"):
        if self.RAM_ONLY: return
        
        if self.cursor == chunk:
            return
        
        self.cursor = chunk

        if to_load == "embedding":
            self.load_embeddings(chunk)

        if to_load == "texts":
            self.load_texts(chunk)

        if to_load == "both":
            self.load_embeddings(chunk)
            self.load_texts(chunk)

    def add(self, strings):
        if self.db_end is None:
            raise Exception("db not initialized")
            
        strings = list(set(strings))

        in_chunk_pos = self.db_end % self.CHUNK_SIZE

        if (len(strings) + in_chunk_pos) > self.CHUNK_SIZE:
            """
            bsp:
                in_chunk_pos = 7
                chunk_size = 10
                len(strings) = 17

                => 3, 13, 23, 27
            """

            pos = self.CHUNK_SIZE - in_chunk_pos
            self.add(strings[:pos])

            while pos < len(strings):
                self.add(strings[pos : pos + self.CHUNK_SIZE])
                pos += self.CHUNK_SIZE

            return

        new_embeddings = model.encode(strings, normalize_embeddings=True)

        self.load_position(self.db_end, "both")

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.concatenate((self.embeddings, new_embeddings), axis=0)

        if self.texts is None:
            self.texts = strings
        else:
            self.texts += strings

        self.save_position(self.db_end, "both")

        self.db_end += len(new_embeddings)

        if not self.RAM_ONLY:
            with open(f"{self.file_name}data.json", "w") as f:
                f.write(str(self.db_end))

    def query(self, query, k=5):
        if self.db_end is None:
            raise Exception("db not initialized")
        
        query_embedding = model.encode(query, normalize_embeddings=True)

        chunk = 0

        all_top_indices = []
        indices_to_text = {}

        while chunk * self.CHUNK_SIZE < self.db_end:
            self.load_chunk(chunk)

            k = min(len(self.embeddings), k)
            
            scores = self.embeddings @ query_embedding  # cosine similarity

            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[
                np.argsort(scores[top_indices])[::-1]
            ]  # top k indices

            if len(all_top_indices) == 0:
                for i in top_indices:
                    indices_to_text[chunk * self.CHUNK_SIZE + i] = self.texts[i]
                    all_top_indices.append([chunk * self.CHUNK_SIZE + i, scores[i]])
            else:
                for i in top_indices:
                    for i_ in range(0, len(all_top_indices)):
                        if all_top_indices[i_][1] < scores[i]:
                            all_top_indices.insert(
                                i_, [chunk * self.CHUNK_SIZE + i, scores[i]]
                            )
                            indices_to_text[chunk * self.CHUNK_SIZE + i] = self.texts[i]

                            del indices_to_text[all_top_indices[-1][0]]
                            del all_top_indices[-1]
                            break

            chunk += 1

        return [indices_to_text[i[0]] for i in all_top_indices]
