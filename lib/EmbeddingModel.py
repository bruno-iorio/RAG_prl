import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('../embedding/embeddings.npy')
ids = np.load('../embedding/ids.npy')
