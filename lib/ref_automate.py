from individual_llm_ref import *

id = 0
query_dic = index_topic("../topics/alltopics.xml")
print(query_dic.keys())

for model in ["llama3.1"]:   
    dic_to_ref = "../topics/reformed/" +model+"/pred_location"
    for i in range(0,50):
        q_id = list(query_dic.keys())[i]
        query = query_dic[q_id]['title']
        desc = query_dic[q_id]['narr']
        id = 0
        ref_store(dic_to_ref,get_point_query,q_id,query,desc,id,model)
        print("model {}, id: {}, ref_id: {} Stored \n".format(model,q_id,id))
