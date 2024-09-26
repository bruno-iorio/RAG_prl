
import re

data_dir = 'data'


topic_path = "topics/alltopics.xml"
def index_topic(topic_path): # Return dict of queries
    dic = dict()
    with open(topic_path,'r',encoding='utf-8') as file:
        filecont = file.read()
        topic_list = re.findall("<top>.*?</top>",filecont,re.DOTALL)
        title,id,desc,narr = "","","",""
        for top in topic_list:
            try:
                title = re.search("<title>.*?</title>",top,re.DOTALL).group().replace("<title>","").replace("</title>","").strip()
            except:
                continue
            try:
                id = re.search("<num>.*?</num>",top,re.DOTALL).group().replace("<num>","").replace("</num>","").strip()
            except:
                continue
            try:
                desc = re.search("<desc>.*?</desc>",top,re.DOTALL).group().replace("<desc>","").replace("</desc>","").strip()
            except: 
                continue
            try: 
                narr = re.search("<narr>.*?</narr>",top,re.DOTALL).group().replace("<narr>","").replace("</narr>","").strip()
            except:
                pass
            try:
                dic[id] = {'title':title, 'desc': desc, 'narr':narr}
            except:
                continue
    return dic

qrels_path = "qrels/alltopics.txt"

def find_relq_dic(file_path): # return dictionary that associate each query to its relevant documents
    dic = dict()
    with open(file_path,'r') as file:
        lines = file.readlines()
        for line in lines:
            x = line.strip().split()
            quer = x[0]
            doc = x[2]
            rel = int(x[3])
            if quer not in dic.keys():
                dic[quer] = []
            if rel == 1:
                dic[quer].append(doc)
        
    return dic

    

