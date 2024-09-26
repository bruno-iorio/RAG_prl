## whoosh imports
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh import index

## sentence transformer to encode documents    
from sentence_transformers import SentenceTransformer

## basic imports
import os
import re
import numpy as np


data_dir = 'data'
analyzer = StemmingAnalyzer()


schema = Schema(
    id=ID(stored=True),
    title=TEXT(stored=True,analyzer=analyzer),
    content=TEXT(stored=True,analyzer=analyzer)
)

index_dir = "index_dir"

os.makedirs(index_dir, exist_ok=True)
ix = create_in(index_dir,schema)

model = SentenceTransformer('all-MiniLM-L6-v2')


def index_files(data_dir,index,model = None):
    documents = []
    ids = []
    filecount = 0 
    writer = index.writer(limitmb=512)  # Increases memory buffer to 512 MB
    for dir in os.listdir(data_dir):
        dir_path = os.path.join(data_dir,dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(dir_path,filename)
                errorcount = 0
                with open(file_path,'r',encoding='iso-8859-1') as file:
                    file_cont = file.read()
                    docs_in_file = re.findall('<DOC>.*?</DOC>',file_cont,re.DOTALL)
                    for docs in docs_in_file:
                        empty = False
                        try:
                            title = re.search('<HEADLINE>.*?</HEADLINE>',docs,re.DOTALL).group().replace("<HEADLINE>","").replace("</HEADLINE>","")
                        except:
                            errorcount += 1
                            title = ""
                        try:
                            content = re.search('<TEXT>.*?</TEXT>',docs,re.DOTALL).group().replace("<TEXT>","").replace("</TEXT>","")
                        except:
                            errorcount += 1
                            content = ""
                            continue
                        try:
                            id = re.search('<DOCNO>.*?</DOCNO>',docs,re.DOTALL).group().replace("<DOCNO>","").replace("</DOCNO>","")
                        except:
                            errorcount += 1
                            id =""
                            continue

                        title = title.strip()
                        content = content.strip()
                        id = id.strip()
                        document = f"{title}\n{content}"
                        documents.append(document)
                        writer.add_document(id=id,title=title,content=content)
                print("{} errors found at {} with {} docs".format(errorcount,filecount,len(docs_in_file)))
                filecount += 1
    print("commiting: ")
    writer.commit()
    print("encoding: ")
    embeddings = model.encode(documents, convert_to_tensor=True)
    return ids, embeddings.cpu()

ids,embeddings = index_files(data_dir,ix,model)

# Save the embeddings and IDs
np.save('embeddings.npy', embeddings)
np.save('ids.npy', np.array(ids))
