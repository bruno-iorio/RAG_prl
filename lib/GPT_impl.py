from openai import OpenAI
from six.moves import cPickle as pickle #for performance
from .TopicIndexer import TopicIndexer
from .CreateDict import create_reformed_dict
import numpy as np
import os
import re
import json

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


def get_transf(text, model="gpt-4o-mini"):
   return client.chat.completions.create(
        model= 'gpt-4o-mini',
        messages=[{'role':'system','content':text}]
    ).choices[0].message.content

def reformulate_gpt(topicpath,savepath):
    topicdic = TopicIndexer.index_topic(topicpath)
    count = 1
    with open(savepath,'a+') as file:
        for query_id in topicdic.keys():
            print(query_id)
            for ref_id in range(3):
                try:
                    ref = get_transf("reformulate the query into an ideal document: {}; given the description: {}; Make it a document and not just a title, with actual information (not necessarily precise) ;I want the text to be just the paragraphs, cointaning introduction, development paragraphs and conclusion, but don't indicate them like 'conclusion:'; don't add any comments before or after; don't use markdown like '*';don't make a list of what an ideal document should contain;".format(topicdic[query_id]['title'],topicdic[query_id]['desc']))
                    if count == 1:
                        count = 0
                        print(ref)
                    file.write("<doc>\n")
                    file.write("<query>{}</query>\n".format(query_id))
                    file.write("<ref_id>{}</ref_id>\n".format(ref_id))
                    file.write("<ref>{}</ref>\n\n".format(ref))
                    file.write("</doc>\n")
                except:
                    print("error at {},{}".format(ref_id,query_id))


def embedd_reformulated(path_ref,jsonl_path):
    ref_dic = create_reformed_dict(path_ref)
    with open(jsonl_path,'w') as file:
        for id in list(ref_dic.keys()):
            for ref_id in list(ref_dic[id].keys()):
                file.write(json.dumps({"custom_id":"{}__{}".format(id,ref_id),
                                    "method":"POST",
                                    "url":"/v1/embeddings",
                                    'body':{"model":'text-embedding-ada-002',
                                            "input":ref_dic[id][ref_id],
                                            "encoding_format":'float'}}
                                    ))
                file.write('\n')


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

class ParserGPT:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.dic = dict()
    
    def endic_files(self,filename,savepath):
        with open(filename,'r') as file:
            for line in file:
                json_obj = json.loads(line)
                self.dic[json_obj['custom_id']] = json_obj['response']['body']['data'][0]['embedding']
        with open(savepath,'wb') as savefile:
            pickle.dump(self.dic,savefile)
            np.save(savepath,self.dic)
    def endic_directory(self,direcpath,savepath):
        for filename in os.listdir(direcpath):
            self.endic_files(direcpath+filename,savepath)

class IndexerGPT:
    def __init__(self):
        self.ids = []
        self.documents = []
        self.json = dict()

    def index_files(self,data_dir,batch_size=10000):
        filecount = 0 
        self.ids = []
        self.documents = []
        temp_list = []
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
                            try:
                                content = re.search('<TEXT>.*?</TEXT>',docs,re.DOTALL).group()
                                content = content.replace("\n", " ")
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
                            content = content.strip()
                            id = id.strip()
                            self.ids.append(id)
                            self.documents.append(content)
                    print("{} errors found at {} with {} docs".format(errorcount,filecount,len(docs_in_file)))
                    filecount += 1
        print("n of ids are: ",len(self.ids))
        print("n of files are: ",(len(self.ids) + batch_size - 1)//batch_size)
        for i in range(1, 1 + (len(self.ids) + batch_size -1) // batch_size):
            if i*batch_size < len(self.ids):
                continue 
            print(i)
            with open("batches_{}.jsonl".format(i),'+w') as file:
                for id, doc in zip(self.ids[(i-1)*batch_size : min(len(self.ids),i*batch_size)],self.documents[(i-1)*batch_size : min(len(self.documents),i*batch_size)]):
                    file.write(json.dumps({"custom_id":id,"method":"POST",
                                        "url":"/v1/embeddings",
                                        'body':{"model":'text-embedding-ada-002',
                                                'input':doc,
                                                'encoding_format':'float'}}
                                        ))
                    file.write('\n')


