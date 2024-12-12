import tensorflow as tf
import numpy as np
import os
import re

from tensorflow.keras import layers, Model

from .search import rag_dense_lat, eval
from .TopoIndexer import TopoIndexer
from .EmbeddingModel import model, embeddings, ids
from .CreateDict import create_reformed_dict
data_dir_topo = "../../geoclef-GNtagged"
topoIndexer = TopoIndexer()
topo_dic = topoIndexer.index_topo(data_dir_topo)


def concat_queries(llm_model,model,embeddings=embeddings,ids=ids,topo_dic=topo_dic):
    ref = create_reformed_dict("../topics/reformed/" + llm_model + "/desc_without_markdown/all_reformulations.txt")
    ref_topo = create_reformed_dict("../topics/reformed/" +llm_model+"/pred_location/all_reformulations.txt")
    relevant = find_relq_dic("../qrels/alltopics.txt")
    X1 = []
    X2 = []
    Y =  []
    for query_id in ref.keys():
        for ref_id in ref[query_id].keys():
            if(int(query_id[-2:]) < 25):
                query_ref = ref[query_id][ref_id]
                coord_str = ref_topo[query_id]['0'].replace('[','').replace(']','')
                coord = coord_str.split(',')
                lat = float(coord[0])
                lon = float(coord[1])
                query_embedding = model.encode(query_ref, convert_to_tensor=True).cpu()
                vec = tf.constant([lat,lon])
                X1.append(query_embedding)
                X2.append(vec)
                y = 0
                ev = 0
                for w in np.linspace(0,1,num=11):
                    res = eval(rag_dense_lat(query_ref,model,embeddings,ids,topo_dic=topo_dic,lat=lat,lon=lon,w=w),relevant[query_id])
                    if res[2] > ev:
                        print(query_id,res[2], w)
                        ev = res[2]
                        y = w
                Y.append(y)
                print("Added was: ",y)
    return X1, X2, Y
def create_model(embedding_dim=384, additional_feature_size=2):
    # Input layers
    sentence_embedding_input = layers.Input(shape=(embedding_dim,), name="sentence_embedding")
    additional_features_input = layers.Input(shape=(additional_feature_size,), name="additional_features")

    # Concatenate Sentence Embedding with Additional Features
    concatenated = layers.Concatenate()([sentence_embedding_input, additional_features_input])

    # Dense layers for processing the combined features
    x = layers.Dense(128, activation='relu')(concatenated)
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with sigmoid activation to produce values between 0 and 1
    output = layers.Dense(1, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=[sentence_embedding_input, additional_features_input], outputs=output)

    return model

# Create and compile the model
#additional_feature_size = 2  # Number of additional features
#embedding_dim = 384  # Sentence Transformer embedding size
#model = create_model(embedding_dim=embedding_dim, additional_feature_size=additional_feature_size)
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
 #             loss='binary_crossentropy', metrics=['accuracy'])
#model.fit([X1,X2],Y,epochs=10)
