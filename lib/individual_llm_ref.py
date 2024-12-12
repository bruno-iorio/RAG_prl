import re
import os
from topics_indexer import index_topic
from create_llm import create_llm

## these are functions that will automatically create reformulated queries so we can use RAG

def get_point_query(query,model,desc):
    llm = create_llm(model)
    point = llm.complete("based on the query: {}; and based on the description: {}; give me only one the latitude and only one longitude of a place that is associated to the place refered in the query; in your output, the latitude must be between <lat> </lat> and the longitude between <lon> and </lon>; It doesn't need to be extremely accurate; don't add extra comments".format(query,desc)).text
    print(point)
    lat = float(re.findall("<lat>.*?</lat>",point,re.DOTALL)[0].replace("<lat>",'').replace("</lat>",''))
    lon = float(re.findall("<lon>.*?</lon>",point,re.DOTALL)[0].replace("<lon>",'').replace("</lon>",''))
    return [lat,lon]
    
def reformulate_title(query,model,desc): # returns reformulated query string
    llm = create_llm(model)
    reformulated = llm.complete("reformulate the query into another query: {}; This reformulation must fullfil: {}; Don't make a document stating what an ideal docuemnt would contain; Make it just a title; don't add comments before or after; just output the new document. Maximum of 10 words".format(query,desc)).text
    return reformulated

def reformulate_desc(query,model,desc): # returns reformulated query string
    llm = create_llm(model)
    reformulated = llm.complete("reformulate the query into an ideal document: {}; given the description: {}; Make it a document and not just a title, with actual information (not necessarily precise) ;I want the text to be just the paragraphs; don't add any comments before or after; don't use markdown like '*'; don't make a list of what an ideal document should contain;".format(query,desc)).text
    return reformulated

def store_reformulation(dir_to_ref,query_id,ref_id,ref):
    if not os.path.exists(dir_to_ref):
        os.makedirs(dir_to_ref)
    path_to_store = dir_to_ref + '/' + 'all_reformulations' + ".txt"
    with open(path_to_store,'a+') as file:
        file.write("<doc>\n")
        file.write("<query>{}</query>\n".format(query_id))
        file.write("<ref_id>{}</ref_id>\n".format(ref_id))
        file.write("<ref>{}</ref>\n\n".format(ref))
        file.write("</doc>\n")

def ref_store(dir_to_ref, ref_f,query_id,query,ref_desc, ref_id, model):
    ref = ref_f(query, model, ref_desc)
    print("query {}, reformulated. ID : {}".format(query,ref_id))
    store_reformulation(dir_to_ref,query_id,ref_id,ref)
