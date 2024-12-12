import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sentence_transformers import SentenceTransformer, util

from .ModelToponym import concat_queries, create_model
from .search import eval_queries
from .search_gpt import eval_gpt_queries

def gpt_plot_topk(queries_dic,ids,embeddings_space,top_k_min,top_k_max): # make sure to use this function after the setup
    name_to_save = "Topk - rag_dense - gpt.png"
    x_axis = range(top_k_min,top_k_max)
    y_axis_prec10, y_axis_recall, y_axis_prec = [], [], []
    for top_k in x_axis:
        prec, recall, prec10 = eval_gpt_queries(queries_dic,ids,embeddings_space)
        y_axis_prec.append(prec)
        y_axis_prec10.append(prec10)
        y_axis_recall.append(recall)
    plt.xlabel("Top K")
    plt.title("Evaluation varying Top K - gpt")
    plt.plot(x_axis,y_axis_prec,label='precision')
    plt.plot(x_axis,y_axis_prec10,label='prec10')
    plt.plot(x_axis,y_axis_recall,label= 'recall')
    plt.savefig(name_to_save)

def gpt_plot_w1(name_to_save,queries_dic,ids,embeddings_space,top_k,w_min,w_max):
    name_to_save = "Ws - rag_dense - gpt.png"
    x_axis = np.linspace(w_min,w_max,num=11)
    y_axis_prec10, y_axis_recall, y_axis_prec = [], [], []
    for w in x_axis:
        prec, recall, prec10 = eval_gpt_queries_lat(queries_dic,ids,embeddings_space,w)
        y_axis_prec.append(prec)
        y_axis_prec10.append(prec10)
        y_axis_recall.append(recall)
    plt.xlabel("weights")
    plt.title("Plot for varying w on {}, with top_k {}".format(method,top_k))
    plt.plot(x_axis,y_axis_prec,label='precision')
    plt.plot(x_axis,y_axis_prec10,label='prec10')
    plt.plot(x_axis,y_axis_recall,label= 'recall')
    plt.savefig(name_to_save)

def plot_topk(method,model,top_k_min,top_k_max,w):
    name_to_save = "Topk - {} - {}.png".format(method,model)
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.set_xlabel("top_k")
    ax.set_title("Evaluation varying Top K - {}".format(method))
    x_axis = range(top_k_min,top_k_max)
    y_axis_prec10, y_axis_recall, y_axis_prec = [], [], []
    for top_k in x_axis:
        prec, recall, prec10 = eval_queries(method,llm_model=model,top_k=top_k,w=w)
        y_axis_prec.append(prec)
        y_axis_prec10.append(prec10)
        y_axis_recall.append(recall)
    ax.plot(x_axis,y_axis_prec,label='precision')
    ax.plot(x_axis,y_axis_prec10,label='prec10')
    ax.plot(x_axis,y_axis_recall,label='recall')
    fig.savefig(name_to_save)
    plt.close(fig)

def plot_w1(method,model,top_k,w_min,w_max):
    name_to_save = "Ws - {} - {}.png".format(method,model)
    x_axis = np.linspace(w_min,w_max,num=11)
    y_axis_prec10, y_axis_recall, y_axis_prec = [], [], []
    for w in x_axis:
        prec, recall, prec10 = eval_queries(method,llm_model=model,top_k=top_k,w=w)
        y_axis_prec.append(prec)
        y_axis_prec10.append(prec10)
        y_axis_recall.append(recall)
    plt.xlabel("weights")
    plt.title("Plot for varying w, with top_k {}".format(method,top_k))
    plt.plot(x_axis,y_axis_prec,label='precision')
    plt.plot(x_axis,y_axis_prec10,label='prec10')
    plt.plot(x_axis,y_axis_recall,label= 'recall')
    plt.savefig(name_to_save)


# plot_w1('RAG_dense_lat','llama3.1',20,0,1)

def predict_nn(method,llm_model,top_k):
    model1 = SentenceTransformer('all-MiniLM-L6-v2')
    X1,X2,Y = concat_queries(llm_model,model1)
    sentence_embeddings_array = np.array([tensor.numpy() for tensor in X1])
    additional_features_array = np.array([tensor.numpy() for tensor in X2])
    labels = np.array([elem for elem in Y]).reshape(-1,1)
    print("Sentence embeddings shape:", sentence_embeddings_array.shape)
    print("Additional features shape:", additional_features_array.shape)
    print("Labels shape:", labels.shape)
    model1 = create_model(384,2)
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.fit([sentence_embeddings_array,additional_features_array],labels,epochs=10,batch_size=16)
    prec, recall, prec10 = eval_queries(method,llm_model=llm_model,top_k=top_k,nn=model1)
    with open("predictions",'w+') as file:
        file.write("prec: {}, recall: {}, prec10: {}".format(prec,recall,prec10))




"""
fig,axs = plt.subplots(3,2,figsize=(12,4))
methods = ['RAG_title_dense','RAG_dense_lat','standart_title','standart_desc','dense_title','RAG_desc_dense']
for i in range(2):
    for j in range(3):
        plot_topk(axs[i,j],methods[3*i + j],'llama3.1',1,40,0.5)
fig.savefig("Varying_topk.png")
"""








