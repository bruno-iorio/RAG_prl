import re
import os
import numpy as np
import torch
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
from whoosh import index

from sentence_transformers import util

from .CreateDict import create_reformed_dict
from .TopicIndexer import TopicIndexer
from .Createllm import create_llm
from .TopoIndexer import TopoIndexer, compute_distance, min_dist
from .EmbeddingModel import model, embeddings, ids
from .GPT_impl import load_dict
from .eval import eval, f1_score

data_dir_topo = "../../geoclef-GNtagged" # geotagged documents
qrels_path = "../qrels/alltopics.txt"    # relevant documents for each query
index_dir = "../index_dir"               # indexed documents
topic_path = "../topics/alltopics.xml"   # queries are stored here
embedding_dir = "../embedding/embeddings.npy"
ids_dir = "../embedding/ids.npy"

# Open the index 
ix = index.open_dir(index_dir)

embeddings = np.load('../embedding/embeddings.npy')
ids = np.load('../embedding/ids.npy')
# creating all the dictionaries
topoIndexer = TopoIndexer()
topo_dic = topoIndexer.index_topo(data_dir_topo)
topicIndexer = TopicIndexer()
query_dic = topicIndexer.index_topic(topic_path)
relevant = topicIndexer.find_relq_dic(qrels_path)

## functions for resolving each search method

def search_dense_title(title, model, embeddings, ids, top_k=15): #function using embeddings but without llm
    query_embedding = model.encode(title, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    res = []
    for idx in top_results:
        res.append(ids[idx])
    return res 
 
def search_dense_desc(query,desc,model, embeddings, ids, top_k=15): #function using embeddings but without llm
    query_embedding = model.encode(query + ". " + desc, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    res = []
    for idx in top_results:
        res.append(ids[idx])
    return res 

def search_standart_title(query_str,ix=ix): # standart tf on title
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query)
        res = []
        for result in results:
            res.append(str(result['id']))
    return res

def search_standart_desc(query_str,desc_str,ix=ix): # stabdart tf on title and desc
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(desc_str)
        results = searcher.search(query)
        res = []
        for result in results:
            res.append(str(result['id']))
    return res

def rag_tf(ref): # with llm and with embeddings -> prob final use
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(ref)
        results = searcher.search(query)
        res = []
        for result in results:
            res.append(str(result['id']))
    return res

def rag_dense(ref,model, embeddings, ids, top_k=15): #function using embeddings but without llm
    query_embedding = model.encode(ref, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    res = []
    for idx in top_results:
        res.append(ids[idx])
    return res 


def rag_dense_lat_nn(ref,model,embeddings,ids,nn,topo_dic=topo_dic,lat=0,lon=0,top_k=15): ## using geotagged info with nn model
    query_embedding = model.encode(ref, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    res_2 = []
    res_dic = dict()
    for idx in top_results:
        id = ids[idx]
        final_score=0
        try:
            centroid = topo_dic[id]['centroid']
            coef = topo_dic[id]['normalized']
            list_of_topo = topo_dic[id]['list']
            min_ = min_dist(list_of_topo,[lat,lon])/30
            if min_ > 1:
                min_ = 1
            dist = compute_distance(centroid,[lat,lon])/30
            if dist > 1:
                dist = 1
            w = nn.predict([query_embedding,[lat,lon]])
            final_score = (1-w)*(coef*(1-dist) + (1-coef)*(1-min_)) + w*cos_scores[idx].item()
        except: 
            final_score = cos_scores[idx].item()
        res_dic[id] = final_score

    res = sorted(res_dic,key=res_dic.get)
    return res[::-1] 



def rag_dense_lat(ref,model,embeddings,ids,topo_dic=topo_dic,lat=0,lon=0,w = 0.8,top_k=15): #using geotagged info
    query_embedding = model.encode(ref, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    top_results = np.argsort(-cos_scores)[:top_k]
    res_2 = []
    res_dic = dict()
    for idx in top_results:
        id = ids[idx]
        final_score=0
        try:
            centroid = topo_dic[id]['centroid']
            coef = topo_dic[id]['normalized']
            list_of_topo = topo_dic[id]['list']
            min_ = min_dist(list_of_topo,[lat,lon])/30
            if min_ > 1:
                min_ = 1
            dist = compute_distance(centroid,[lat,lon])/30
            if dist > 1:
                dist = 1
            final_score = (1-w)*(coef*(1-dist) + (1-coef)*(1-min_)) + w*cos_scores[idx].item()
        except: 
            final_score = cos_scores[idx].item()
        res_dic[id] = final_score
    res = list(dict(sorted(res_dic.items(), key=lambda item: item[1])).keys())
    return res[::-1]
## -----------------------------
## single query evaluation 
## systematic query evaluation
def eval_queries(method,save_name,llm_model='llama3.1',precAt=None,top_k=15,w=0.8,nn=None):
    global embeddings, ids
    global query_dic
    global relevant
    if method == "standart_title":
        avg = [0,0,0]
        i = 0
        for query_id in query_dic.keys():
            title = query_dic[query_id]['title']
            pred = search_standart_title(title)
            ans = relevant[query_id]
            if len(ans) == 0 or precAt > len(pred):
                continue
            precision, recall, prec10 = eval(pred,ans,precAt)
            avg[0] += precision
            avg[1] += recall
            avg[2] += prec10
            i += 1
            print(avg)
        avg[0] = avg[0] / i
        avg[1] = avg[1] / i
        avg[2] = avg[2] / i

    elif method == "standart_desc":
        avg = [0,0,0]
        i = 0
        for query_id in query_dic.keys():
            title = query_dic[query_id]['title']
            desc = query_dic[query_id]['narr']
            print(desc)
            pred = search_standart_desc(title,desc)
            ans = relevant[query_id]
            if len(ans) == 0:
                continue
            precision, recall, prec10 = eval(pred,ans,precAt)
            avg[0] += precision
            avg[1] += recall
            avg[2] += prec10
            i += 1
            print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "dense_title":
        avg = [0,0,0]
        i = 0
        for query_id in query_dic.keys():
            title = query_dic[query_id]['title']
            pred = search_dense_title(title,model,embeddings,ids,top_k=top_k)
            ans = relevant[query_id]
            if len(ans) == 0:
                continue
            precision, recall, prec10 = eval(pred,ans,precAt)
            avg[0] += precision
            avg[1] += recall
            avg[2] += prec10
            i += 1
            print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "dense_desc":
        avg = [0,0,0]
        i = 0
        for query_id in query_dic.keys():
            title = query_dic[query_id]['title']
            desc = query_dic[query_id]['narr']
            pred = search_dense_desc(title,desc,model,embeddings,ids,top_k=top_k)
            ans = relevant[query_id]
            if len(ans) == 0:
                continue
            precision, recall, prec10 = eval(pred,ans,precAt)
            avg[0] += precision
            avg[1] += recall
            avg[2] += prec10
            i += 1
            print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "RAG_title_tf":
        ref = create_reformed_dict("../topics/reformed/" + llm_model + "/title2/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in ref.keys():
            for ref_id in ref[query_id].keys():
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_tf(reformulated)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "RAG_desc_tf":
        ref = create_reformed_dict("../topics/reformed/" +llm_model + "/desc_without_markdown/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in ref.keys():
            for ref_id in ref[query_id].keys():
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_tf(reformulated)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "RAG_title_dense":
        ref = create_reformed_dict("../topics/reformed/" + llm_model+"/desc3/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in ref.keys():
            for ref_id in ref[query_id].keys():
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_dense(reformulated,model,embeddings,ids,top_k=top_k)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "RAG_dense_lat":
        ref = create_reformed_dict("../topics/reformed/"+llm_model+"/alltopics.xml")
        ref_topo = create_reformed_dict("../topics/reformed/llama3.1/pred_location/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in list(ref.keys())[:50]:
            for ref_id in ref[query_id].keys():
                coord_str = ref_topo[query_id]['0'].replace('[','').replace(']','')
                coord = coord_str.split(',')
                lat = float(coord[0])
                lon = float(coord[1])
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_dense_lat(reformulated,model,embeddings,ids,lat=lat,lon=lon,w=w,top_k=top_k)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i

    elif method == "RAG_dense_lat_nn":
        ref = create_reformed_dict("../topics/reformed/"+llm_model+"/desc_without_markdown/all_reformulations.txt")
        ref_topo = create_reformed_dict("../topics/reformed/llama3.1/pred_location/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in list(ref.keys())[:50]:
            for ref_id in ref[query_id].keys():
                coord_str = ref_topo[query_id]['0'].replace('[','').replace(']','')
                coord = coord_str.split(',')
                lat = float(coord[0])
                lon = float(coord[1])
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_dense_lat_nn(reformulated,model,embeddings,ids,nn,lat=lat,lon=lon,top_k=top_k)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i
    elif method == "RAG_desc_dense":
        ref = create_reformed_dict("../topics/reformed/"+llm_model+"/desc_without_markdown/all_reformulations.txt")
        avg = [0,0,0]
        i = 0
        for query_id in ref.keys():
            for ref_id in ref[query_id].keys():
                i += 1
                reformulated = ref[query_id][ref_id]
                pred = rag_dense(reformulated,model,embeddings,ids,top_k=top_k)
                ans = relevant[query_id]
                if len(ans) == 0:
                    continue
                precision, recall, prec10 = eval(pred,ans,precAt)
                avg[0] += precision
                avg[1] += recall
                avg[2] += prec10
                print(avg)
        avg[0] = avg[0]/i
        avg[1] = avg[1]/i
        avg[2] = avg[2]/i
    f1 = f1_score(avg[0],avg[1])
    with open(save_name,'a+') as file:

        file.write(method + '\n')
        file.write(llm_model + '\n')
        if precAt is not None:
            file.write(f"precAt {precAt}")
        file.write(f"prec: {avg[0]}\n")
        file.write(f"recall: {avg[1]}\n")
        file.write(f"prec@10: {avg[2]}\n")
        file.write(f"f1_score: {f1}\n\n")

    return avg

