import re
import os
import numpy as np

#imports from whoosh
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser
from whoosh import index

# For encoding and retrieval
from sentence_transformers import SentenceTransformer, util
from topics_indexer import find_relq_dic, index_topic
#create llm fast
from llm_ref import create_llm

qrels_path = "qrels/alltopics.txt"  # relevant documents for each query
index_dir = "index_dir"             # indexed documents
topic_path = "topics/alltopics.xml" # queries are stored here
# Open the index
ix = index.open_dir(index_dir)
# Function to search the index

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.load('embeddings.npy')
ids = np.load('ids.npy')

# Function to perform a search
def search(query, model, embeddings, ids, top_k=5): #function using embeddings but without llm
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities between the query and all document embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    
    # Get the top-k highest scores
    top_results = np.argsort(-cos_scores)[:top_k]
    
    for idx in top_results:
        print(f"ID: {ids[idx]}")
        print(f"Score: {cos_scores[idx]:.4f}")
        with open(os.path.join(data_directory, ids[idx]), 'r', encoding='utf-8') as file:
            print(file.read())
            print("\n" + "="*50 + "\n")

def search_index(query_str,desc = None,llm = None): # function not using embeddings
    with ix.searcher() as searcher:
        if llm is not None:
            try: 
                query_str = llm.complete("reformulate this query into an ideal document using the query {} given the description {}. Start with <ref> and end with </ref>. Just output the document, nothing else".format(query_str,desc))
                query_str = query_str.text
                query_str = re.search("<ref>.*?</ref>",query_str,re.DOTALL).group().replace("<ref>",'').replace("</ref>","")
            except:
                pass
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query)
        res = []
        for result in results:
            res.append(str(result['id']))
    return res

def search_val(query,desc,model, embeddings, ids): # with llm and with embeddings -> prob final use
    # create llm and reformulate;
    llm = create_llm()
    query = llm.complete("reformulate the query {} into an ideal document based on the description {}. Don't include: metadata , like ids, title; Don't divide in sections; Don't add any comment about the document in the beginning or in the end of it;".format(query,desc)).text
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarities between the query and all document embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding.cpu(), embeddings)[0]
    # Get the top-k highest scores
    top_results = np.argsort(-cos_scores)
    result = []
    for idx in top_results:
        if cos_scores[idx] > 0.5: # test with different parameters later.
            result.append(ids[idx])
    return result

# To be done: organize this function
def eval_search(model,embeddings,ids):
    topic_path = "topics/alltopics.xml"
    qrels_path = "qrels/alltopics.txt"
    qrels_dic = find_relq_dic(qrels_path) #return dictionary with 
    topic_dic = index_topic(topic_path) #dictionary where you have an id and you receive the query for the given id
    av_prec = 0
    numb = 0
    avg_prec = 0
    avg_prec_10 = 0
    avg_recall = 0
    for query_id in topic_dic.keys():
        query_description = topic_dic[query_id]['desc']
        query_title = topic_dic[query_id]['title']
        relevant_docs = qrels_dic[query_id] 
        print("Analysing Query: {}\n There is a total of {} relevant documents \n".format(query_title,len(relevant_docs)))
        
        pred = search_val(query_title,query_description,model,embeddings,ids) #list with ids of predicted docuemtns
        print("len pred :",len(pred))
        
        total_relevant = len(relevant_docs)
        correct = 0
        correctat10 = 0
        precision = None
        precision_at10 = None
        recall = None
        
        i = 0
        for predicted in pred:
            if predicted in relevant_docs:
                correct += 1
            if i < 10:
                correctat10 += 1
            i += 1
        if total_relevant == 0 and len(pred) == 0:
            precision = 1
            recall    = 1 
            precision_at10 = 1
        elif total_relevant == 0 and len(pred) != 0:
            precision = 0
            recall    = 0 
            precision_at10 = 0
        elif total_relevant != 0 and len(pred) == 0:
            precision = 0
            recall    = 0 
            precision_at10 = 0
        else: 
            precision = correct/len(pred)
            recall = correct/total_relevant
            precision_at10 = correctat10/10
        print("Precision: {} \nPrecision@10: {} \nRecall {} \n".format(precision,precision_at10,recall))
        avg_prec += precision
        avg_prec_10 += precision_at10
        avg_recall += recall
    avg_prec = avg_prec/len(topic_dic.keys())
    avg_prec_10 = avg_prec_10/len(topic_dic.keys())
    avg_recall = avg_recall/len(topic_dic.keys())
    
    return avg_prec,avg_prec_10,avg_recall

prec, prec10, avg_recall = eval_search(model,embeddings,ids)
print(f"avg_prec: {prec}\n avg_prec_10: {prec10}\n avg_recall: {avg_recall}")













