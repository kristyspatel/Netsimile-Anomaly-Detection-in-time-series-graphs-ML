__author__ = 'Kristy'
import time
from os import listdir
from os.path import join
from igraph import  *
import scipy as sc
from scipy import stats
import numpy as np
import sys
import scipy.spatial.distance
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *

""" find the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: [list of neighbours]}
    e.g.  {1:[0,2] , 2:[1,3]}
"""
def get_egonet (g):
    egonet = {k.index: g.neighbors(k,mode=ALL) for k in g.vs};
    return egonet;

""" To find the degree of nodes in the given graph, Return value will be a dictionary {vertex_index : degree}
    e.g. {1:2 , 2:2 }
"""
def get_di(g):
    neighbor_size = {k.index : g.degree(k,mode=ALL) for k in g.vs};
    return neighbor_size;

""" To find the clustering index of the nodes in the given graph. Return value will be a dictionary {vertex_index: clustering_index}
"""
def get_ci(g):
    clustering_index = {k.index : g.transitivity_local_undirected(vertices=k,mode=TRANSITIVITY_ZERO) for k in g.vs};
    return clustering_index;

""" To find the average number of two hop neighbors of all nodes in the given graph. Return value will be a dictionary {vertex_index: average_two_hop_neighbors}
"""
def get_dni(g):
    two_hop_neighbors = {};
    for key,value in get_egonet(g).iteritems():
        avg_hop = mean([g.degree(k,mode=ALL) for k in value]);
        two_hop_neighbors[key] = avg_hop;
    return two_hop_neighbors;

""" To find the average clustering coefficient of all nodes in the given graph. Return value will be a dictionary {vertex_index: average_clustering coefficient}
"""
def get_cni(g):
    avg_ci = {}
    ci = get_ci(g)
    for key,value in get_egonet(g).iteritems():
        temp = mean([ci[k] for k in value])
        avg_ci[key] = temp
    return avg_ci

""" To find the number of edges in the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: edges_in_egonet}
"""
def get_eegoi(g):
    egonet = get_egonet(g);
    eegoi = {};
    for vertex in g.vs:
        sg = g.subgraph(egonet[vertex.index] + [vertex.index]);
        egonet_es = [(k.source,k.target) for k in sg.es]
        eegoi[vertex.index] = len(egonet_es);
    return eegoi;

""" To find the number of edges going out from the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: outgoing_edges_from_egonet}
"""
def get_eoegoi(g):
    egonet = get_egonet(g);
    eoegoi = {};
    for vertex in g.vs:
        total_vs = [vertex.index];
        for k in egonet[vertex.index]:
            total_vs = total_vs + egonet[k] + [k];
        total_vs = list(set(total_vs));
        sg = g.subgraph(total_vs);
        total_es = [(k.source,k.target) for k in sg.es];
        sg_egonet = g.subgraph(egonet[vertex.index] + [vertex.index]);
        egonet_es = [(k.source,k.target) for k in sg_egonet.es];
        eoegoi[vertex.index] = len(list(set(total_es) - set(egonet_es)));
    return eoegoi;

""" To find the number of neighbors of the egonet of each node in the given graph. Return value will be a dictionary {vertex_index: neighbors_of_egonet}
"""
def get_negoi(g):
    egonet = get_egonet(g);
    negoi = {};
    for vertex in g.vs:
        egonet_vs = [vertex.index] + egonet[vertex.index];
        total_vs = [];
        for k in egonet[vertex.index]:
            total_vs = total_vs +egonet[k];
        total_vs = list(set(total_vs));
        total_vs = [i for i in total_vs if i not in egonet_vs];
        negoi[vertex.index] = len(total_vs);
    return negoi;

""" extract the features of each node in the given graph. Return value will be list of tuples of all features of each node
    e.g. if there are k nodes in graph then return value will be
    [(di0,di0,dni0,cni0,eego0,eoego0,negoi0),(di1,di1,dni1,cni1,eego1,eoego1,negoi1) ... (dik-1,dik-1,dnik-1,cnik-1,eegok-1,eoegok-1,negoik-1)]
"""
def get_features(g):
    di= get_di(g);
    ci= get_ci(g);
    dni= get_dni(g);
    cni=get_cni(g);
    eego=get_eegoi(g);
    eoego=get_eoegoi(g);
    negoi=get_negoi(g);
    all_features = [(di[v.index],ci[v.index],dni[v.index],cni[v.index],eego[v.index],eoego[v.index],negoi[v.index]) for v in g.vs];

    return all_features;

""" Get the signature vector of the graph. Return value will be a list of 35 values
    e.g [mn(f0),md(f0),std_dev(f0),skw(f0),krt(f0), ... mn(f6),md(f6),std_dev(f6),skw(f6),krt(f6)]
"""
def get_signature(g):
    all_features = get_features(g)
    num_nodes = len(all_features);
    signature = [];
    for k in range(0,7):
        feat_agg = [all_features[i][k] for i in range(0,num_nodes)];
        mn = mean(feat_agg);
        md = median(feat_agg);
        std_dev = np.std(feat_agg);
        skw = stats.skew(feat_agg);
        krt = stats.kurtosis(feat_agg);
        signature = signature + [mn,md,std_dev,skw,krt];
    del all_features;
    return signature;

""" find canberra distance between two signature vectors """
def get_canberra_distance(sign1,sign2):
    return abs(scipy.spatial.distance.canberra(sign1, sign2));

""" calculate threshold. two methods used. Method 1: median + 3 * range_mean, Method 2: mean + 3*sigma_c/sqrt(window size = 2)"""
def calculate_threshold(distances):
    n = 2
    moving_range = [abs(distances[i] - distances[i+1]) for i in range(0,len(distances)-1)];
    #range_mean = mean(moving_range);
    range_mean = sum(moving_range)/(len(moving_range)-1);
    med = median(distances)
    UCL = med + 3*range_mean
    """ threshold calculation method 2. uncomment the code below to find threshold by method2 """
    """
    dist_mean = mean(distances);
    sigma_c = range_mean / 1.128;
    UCL = dist_mean + (3 * (sigma_c/sqrt(n)));
    """
    return UCL;

""" determine anomalies on the basis of threshold"""
def anomalies(distances,u_threshold):
    anomalies = []
    for i in xrange(0, len(distances)-1):
        if distances[i] >= u_threshold and distances[i+1] >= u_threshold:
            anomalies.append(i+1);
    return anomalies;


def plot_dist(dists, u_threshold,filename):
    """
    Plot the (N-1) canberra distances comparing each graph with the previous
    """
    figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(dists, "-o")
    axhline(y=u_threshold, ls='-', c='r',
            label='Threshold: $median + 3 * range mean = %0.2f$'%(u_threshold),
            lw=2)
    plt.grid(True)
    plt.legend(loc='best')
    plt.title("Anomaly Detection: ")
    plt.xlabel("Time Series Graphs")
    plt.ylabel("Canberra Distance")
    savefig(join('graph', filename+".png"),bbox_inches='tight')


""" This functions gives the dictionary of the text file as key and the graph object formed by igraph from the text file
    E.g. {'248_autonomous.txt': <igraph.Graph object at 0x000000001230B7D6D8>, '251_autonomous.txt':
    <igraph.Graph object at 0x000000001277D7C8>}
"""
def get_graphs(dir_path):
    file_paths = {f: join(dir_path,f) for f in listdir(dir_path)}
    graphs = {}
    for file,path in file_paths.iteritems():
        try:
            fi = open(path,'r')
            v, e = fi.next().split()
            e_list = [(int(line.split()[0]),int(line.split()[1])) for line in list(fi)]
            g = Graph()
            g.add_vertices(int(v))
            g.add_edges(e_list)
            graphs[file] = g
        finally:
            fi.close()
    return graphs

"""
    run the program like this: python Netsimile.py C:/Users/Kristy/PycharmProjects/Netsimile/input_files
"""
if __name__ == "__main__":
    dir_path = sys.argv[1]
    start_time = time.time()
    """ Read the graphs from files """
    graphs = get_graphs(dir_path)
    """ Ordered graph keys on the basis of time points """
    ordered_graphs = sorted(graphs.keys(),key = lambda k:int(k.split('_',1)[0]))
    """ Obtain graph signatures """
    graph_signatures = {k:get_signature(graphs[k]) for k in ordered_graphs }
    """ Calculate Canberra distance between consecutive time points """
    dists = [get_canberra_distance(graph_signatures[ordered_graphs[i]],
              graph_signatures[ordered_graphs[i-1]]) for i in range(1,len(ordered_graphs))]
    """ Obtain the Upper Threshold """
    u_limit = calculate_threshold(dists)
    """ plot the distances and threshold value """
    plot_dist(dists,u_limit,sys.argv[2])
    """Find anomalous graphs based on threshold """
    anomalous_graphs = anomalies(dists,u_limit)
    with open(join('anomalies',sys.argv[2]+'.txt'), 'w') as fo:
        fo.write("Threshold: "+str(u_limit)+' '+'\n')
        fo.write ("Number of anomalies: "+str(len(anomalous_graphs))+' '+'\n')
        for a in anomalous_graphs:
            fo.write(str(a)+' '+'\n')
    print time.time() - start_time, "seconds"
    print "u_limit",u_limit,"\n"
    print anomalous_graphs, "\n"