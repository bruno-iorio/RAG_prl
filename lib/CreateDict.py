import re

def create_reformed_dict(pathname):
    dic = dict()
    with open(pathname,"r") as file:
        text = file.read()
        lst = re.findall("<doc>.*?</doc>",text,re.DOTALL)
        for doc in lst:
            doc = doc.replace("<doc>","").replace("</doc>","").strip()
            query = re.search("<query>.*?</query>",doc,re.DOTALL).group().replace("<query>","").replace("</query>","").strip()
            if query not in dic.keys():
                dic[query] = dict()
            query_id = re.search("<ref_id>.*?</ref_id>",doc,re.DOTALL).group().replace("<ref_id>","").replace("</ref_id>","").strip()
            ref = re.search("<ref>.*?</ref>",doc,re.DOTALL).group().replace("<ref>","").replace("</ref>","").strip()
            dic[query][query_id] = ref
    return dic
