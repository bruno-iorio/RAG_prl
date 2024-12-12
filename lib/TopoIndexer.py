import os
import re
import numpy as np

# These are the functions to compute distances and other parameters that can be useful for prediction
def compute_distance(centroid, point):
    dist = np.sqrt((centroid[0]-point[0])**2 + (centroid[1] - point[1])**2)
    return dist

def compute_var(list_of_topo,centroid):
    var = 0
    for i in list_of_topo:
        var += compute_distance(centroid,i)**2
    return var/len(list_of_topo)

def euler_normalize(x):
    return 1 + (-1/(1+np.exp(-(4*x-160))))

def min_dist(list_of_topo, x):
    m = np.inf
    for i in list_of_topo:
        m = min(compute_distance(i,x),m)
    return m

# Create a dict with all documents (tagged with toponyms) indexed
class TopoIndexer:
    def __init__(self,topo_dict=None):
        self.topo_dict = topo_dict
    def index_topo(self,data_dir_topo):
        filecount = 0
        topo_dict = dict()
        for filename in os.listdir(data_dir_topo):
            filecount += 1
            if filename.endswith('.xml'):
                with open(data_dir_topo + "/" + filename,'r',encoding='utf-8') as file:
                    doc_count = 0
                    errorcount = 0
                    file_cont = file.read()
                    docs_in_file = re.findall('<DOC>.*?</DOC>',file_cont,re.DOTALL)
                    for docs in docs_in_file:
                        doc_count += 1
                        try:
                            id = re.search('<DOCNO>.*?</DOCNO>',docs,re.DOTALL).group().replace("<DOCNO>","").replace("</DOCNO>","")
                        except:
                            errorcount += 1
                            id =""
                            continue
                        try:
                            toponims = re.findall('<TOPONYM .*?>.*?</TOPONYM>',docs,re.DOTALL)
                            centroid_lat = 0
                            centroid_lon = 0
                            list_of_topo = []
                            for top in toponims:
                                lat = re.search('lat=".*?"',top,re.DOTALL).group().replace("lat=\"","").replace("\"",'')
                                lon = re.search('lon=".*?"',top,re.DOTALL).group().replace("lon=\"","").replace("\"",'')
                                list_of_topo.append([float(lat),float(lon)])
                                centroid_lat += float(lat)
                                centroid_lon += float(lon)
                            centroid = [centroid_lat/len(toponims),centroid_lon/len(toponims)]
                            var = compute_var(list_of_topo,centroid)
                        except:
                            errorcount += 1
                            continue
                        id = id.strip()
                        topo_dict[id] = {'centroid': centroid, 'variance' : var, 'normalized' : euler_normalize(var),'list' : list_of_topo} 
        self.topo_dict = topo_dict
        return topo_dict

