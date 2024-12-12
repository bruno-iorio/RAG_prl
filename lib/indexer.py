## whoosh imports
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh import index
from sentence_transformers import SentenceTransformer
import os
import re
import numpy as np

class Indexer:
    def __init__(self):
        self.analyzer = StemmingAnalyzer()
        self.schema = Schema(
            id=ID(stored=True),
            title=TEXT(stored=True,analyzer=analyzer),
            content=TEXT(stored=True,analyzer=analyzer)
        )
        self.index_dir = "../index_dir"
        os.makedirs(self.index_dir, exist_ok=True)
        self.ix = create_in(self.index_dir,self.schema)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def index_files(self,data_dir):
        filecount = 0 
        writer = self.index.writer(limitmb=512)  # Increases memory buffer to 512 MB
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
                                title = re.search('<HEADLINE>.*?</HEADLINE>',docs,re.DOTALL).group()
                                title.replace("<HEADLINE>","").replace("</HEADLINE>")
                            except:
                                errorcount += 1
                                title = ""
                            try:
                                content = re.search('<TEXT>.*?</TEXT>',docs,re.DOTALL).group()
                                content = content.replace("<TEXT>","").replace("</TEXT>","")
                            except:
                                errorcount += 1
                                content = ""
                                continue
                            try:
                                id = re.search('<DOCNO>.*?</DOCNO>',docs,re.DOTALL).group()
                                id = id.replace("<DOCNO>","").replace("</DOCNO>","")
                            except:
                                errorcount += 1
                                id =""
                                continue

                            title = title.strip()
                            content = content.strip()
                            id = id.strip()
                            #ids.append(id)
                            #documents.append(f"{content}") 
                            writer.add_document(id=id,title=title,content=content)
                    print("{} errors found at {} with {} docs".format(errorcount,filecount,len(docs_in_file)))
                    filecount += 1
        print("commiting: ")
        self.writer.commit()
        print("encoding: ")
        # embeddings = model.encode(documents, convert_to_tensor=True)
        # return #ids # , embeddings.cpu()
#index_files(data_dir,ix)#,model)

# Save the embeddings and IDs
# np.save('../embedding/embeddings.npy', embeddings)
# np.save('../embedding/ids.npy', np.array(ids))
