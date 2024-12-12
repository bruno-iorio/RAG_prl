import torch
import numpy as np 
import json
from .TopicIndexer import TopicIndexer
from .TopoIndexer import TopoIndexer, compute_distance, min_dist
from .CreateDict import create_reformed_dict
from .GPT_impl import load_dict

from .eval import eval,f1_score
def get_output(path):
    dic = dict()
    with open(path,'r') as file:
        for line in file:
            json_obj = json.loads(line)
            query_ref_id = json_obj['custom_id'].split('__')
            query_id = query_ref_id[0]
            ref_id = query_ref_id[1]
            emb = json_obj['response']['body']['data'][0]['embedding']
            if query_id not in dic.keys():
                dic[query_id] = dict()
            dic[query_id][ref_id] = np.array(emb)
    return dic
def setup_all():
    print("loading dict")
    embedding_dict = load_dict("../embedding/gpt_embeddings.pkl")
    print("dict loaded")
    print("transforming ids into list")
    emb_ids = list(embedding_dict.keys())
    print("ids ok")
    print("embeddings_space into list")
    embeddings_space = np.array(list(embedding_dict.values()))
    print(" ok")
    return emb_ids,embeddings_space


def compute_cos_sim(embeddings_space, vector):
    return np.dot(embeddings_space,vector.T).flatten()

def rag_gpt(embedded_query, ids, embeddings_space,topk=15):
    cossim = compute_cos_sim(embeddings_space,embedded_query)
    sorted_indices = np.argsort(-cossim)
    pred = []
    for i in sorted_indices[:topk]:
        pred.append(ids[i])
    return pred

def rag_gpt_lat(embedded_query, ids, embeddings_space, topo_dic, w,lat,lon, topk=15):
    cos_scores = compute_cos_sim(embeddings_space,embedded_query)
    sorted_indices = np.argsort(-cos_scores)
    res_dic = dict()
    for idx in sorted_indices[:topk]:
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

def eval_gpt_queries(queries_dic, ids, embeddings_space,save_name,precAt=None,qrels_path='../qrels/alltopics.txt'):
    topicIndexer = TopicIndexer()
    relevant = topicIndexer.find_relq_dic(qrels_path)
    average = [0,0,0]
    i=0
    for query_id in queries_dic.keys():
        for ref_id in queries_dic[query_id].keys():
            query_emb = np.array(queries_dic[query_id][ref_id])
            pred = rag_gpt(query_emb,ids,embeddings_space)
            ans = relevant[query_id]
            if len(ans) == 0:
                continue
            prec, recall , prec10 = eval(pred,ans,precAt=precAt)
            average[0] += prec
            average[1] += recall
            average[2] += prec10
            print(average)
            i+=1 
    average[0] = average[0] / i
    average[1] = average[1] / i
    average[2] = average[2] / i
    f1 = f1_score(average[0],average[1])
    with open(save_name,'a+') as file:
        file.write('RAG_desc_dense\n')
        file.write('gpt\n')
        file.write(f"prec: {average[0]}\n")
        file.write(f"recall: {average[1]}\n")
        file.write(f"prec@10: {average[2]}\n")
        file.write(f"f1_score: {f1}\n\n")


    return average

def eval_gpt_queries_lat(queries_dic,ids,embeddings_space,save_name,precAt=None,w=0.8,qrels_path='../qrels/alltopics.txt',data_dir_topo = "../../geoclef-GNtagged"):
    topoIndexer = TopoIndexer()
    topicIndexer = TopicIndexer()
    ref_topo = create_reformed_dict("../topics/reformed/llama3.1/pred_location/all_reformulations.txt")
    relevant = topicIndexer.find_relq_dic(qrels_path)
    topo_dic = topoIndexer.index_topo(data_dir_topo)
    average = [0,0,0]
    i=0
    for query_id in list(queries_dic.keys())[:50]:
        for ref_id in queries_dic[query_id].keys():
            coord_str = ref_topo[query_id]['0'].replace('[','').replace(']','')
            coord = coord_str.split(',')
            lat = float(coord[0])
            lon = float(coord[1])
            query_emb = np.array(queries_dic[query_id][ref_id])
            pred = rag_gpt_lat(query_emb,ids,embeddings_space,topo_dic,w,lat,lon)
            ans = relevant[query_id]
            if len(ans) == 0:
                continue
            prec, recall , prec10 = eval(pred,ans,precAt=precAt)
            average[0] += prec
            average[1] += recall
            average[2] += prec10
            print(average)
            i+=1 
    average[0] = average[0] / i
    average[1] = average[1] / i
    average[2] = average[2] / i
    f1 = f1_score(average[0],average[1])
    with open(save_name,'a+') as file:
        file.write('RAG_dense_lat\n')
        file.write('gpt\n')
        file.write('w = .8')
        file.write(f"prec: {average[0]}\n")
        file.write(f"recall: {average[1]}\n")
        file.write(f"prec@10: {average[2]}\n")
        file.write(f"f1_score: {f1}\n\n")


    return average
