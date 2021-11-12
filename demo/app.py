import os
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter


from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

language = ''
config = ''

@app.route('/api/characterization/<string:group>')
def charaterization(group):
    views = config.split('-')
    clustering_algo = views[0]

    inData = views[1].split('_')

    group = int(group)
    df = pd.read_csv(os.path.join('labels', language, clustering_algo+'_'+'_'.join(inData)+'.csv'), index_col=False)
    print(language, config)
    grouped_df =  df[df['cluster_pred']==group]
    
    head = list((i for (i, v) in Counter(grouped_df['label_mix']).most_common()))
    ngrams = get_ngrams(grouped_df)
    
    return {'head':head[0], 'ngrams':ngrams}


def sorting(lst)  -> list:
    lst.sort(key=len)
    return lst

def join_tuple_string(strings_tuple) -> str:
   return ' '.join(strings_tuple)

def string_set(string_list):
    bigrams = list((i for i in string_list if len(i.split()) == 2))
    trigrams = list((i for i in string_list if len(i.split()) == 3))
    
    #bigrams = list(map(join_tuple_string, bigrams))
    #trigrams = list(map(join_tuple_string, trigrams))

    duplicates = list((b for b in bigrams if any(b in string for string in trigrams)))
    
    ngrams = list((x for x in bigrams if x not in duplicates)) + trigrams
    #ngrams = list((tuple(x.split()) for x in ngrams))

    return ngrams

def cleaning(doc):
    import re
    to_del = ['«','»']
    for e in to_del:
        doc = re.sub(e, ' ', doc, flags=re.MULTILINE)
    doc=re.sub(r'\b\w{1,2}\b', '', doc) 
    # remove multiple whitespaces
    doc=re.sub(r'  +', " ", doc)
    return doc


# def most_common(instances):
#     """Returns a list of (instance, count) sorted in total order and then from most to least common"""
#     from operator import itemgetter

#     return sorted(sorted(Counter(instances).items(), key=itemgetter(0)), key=itemgetter(1), reverse=True)

def get_ngrams(df, threshold = 0.001, context_window = 2, top_n=10):
    from nltk.util import ngrams
    from nltk.util import everygrams
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    df['clean_tw'] = df['clean_tw'].apply(lambda x: cleaning(x))
    
    # ngrams_list_o = list((list(everygrams(i.split(), min_len=2, max_len=3)) for i in df.clean_tw.values))
    # ngrams_list = [item for sublist in ngrams_list_o for item in sublist]

    # frequencies = most_common(ngrams_list)
    # percentages = [(instance, count / len(ngrams_list)) for instance, count in frequencies]
    # filtered_bgrams = [instance for instance, count in percentages if count >= threshold]
    
    # ngrams = string_set(sorting(filtered_bgrams))
    # return list((' '.join(k) for k, v in dict(frequencies).items() if k in ngrams))
    corpus = []
    tfidf = TfidfVectorizer(ngram_range=(2, 3))
    
    for name, grouped_df in df.groupby(['cluster_pred']):
        corpus.append(' '.join(grouped_df['clean_tw'].values))
        tfs = tfidf.fit_transform(corpus)
        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(tfs.toarray()).flatten()[::-1]
        
        for r in tfs.toarray():
            ngrams = list((feature_array[i] for i in np.argpartition(r, -top_n)[-top_n:]))
            ngrams_set = string_set(ngrams)
            idx_sup = (len(r)-top_n)
            idx_inf = (len(r)-top_n)-(top_n-len(ngrams_set))
            
            while len(ngrams_set) < top_n:
                ngrams = list((feature_array[i] for i in np.argpartition(r, -(len(ngrams_set)-(len(r)-top_n)))[idx_inf: idx_sup]))
                ngrams = ngrams_set + ngrams
                ngrams_set = list(set(string_set(ngrams)))
                idx_sup = idx_inf
                idx_inf = idx_sup -(top_n-len(ngrams_set))
    return ngrams_set

def get_archive(file_path):
    import zipfile

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_mat)

@app.route('/api/<string:lang>/<string:views>')
def graph_generator(lang, views):
    import glob
    global language
    global config

    config = views
    
    if lang == 'en':
        k_clusters=26
        language = lang
        goldFile='labels/en_gold.csv'
    else:
        k_clusters=16
        language = lang
        goldFile='labels/fr_gold.csv'
    
    tw_db = pd.read_csv(goldFile, index_col=False)
    views = views.split('-')
    clustering_algo = views[0]

    dir_mat = 'matrices/'+language+'/'
    
    matrices = []
    matrices_type = [] #False feature set / True feature set

    inData = views[1].split('_')
    view_type = 'im'
    
    if 'bert' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'im'
        get_archive(dir_mat+'EM-*bert.tar.gz')
        filename = glob.glob(dir_mat+'EM-*bert.csv')[0]
        data = pd.read_csv(filename,index_col=False,header=None)
        DX1 = data.values
        matrices.append(DX1)
        matrices_type.append(False)

    if 'use' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'vad'
        get_archive(dir_mat+'EM-*use.tar.gz')
        filename = glob.glob(dir_mat+'EM-*use.csv')[0]
        data = pd.read_csv(filename, index_col=False,header=None)
        DX2 = data.values
        matrices.append(DX2)
        matrices_type.append(False)

    if 'proB' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'pro'
        get_archive(dir_mat+'ProEM-*bert.tar.gz')
        filename = glob.glob(dir_mat+'ProEM-*bert.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX3 = np.array(data)
        DXX3 = 1 - DXX3; DXX3 = DXX3 - np.diag(np.diag(DXX3))
        matrices.append(DXX3)
        matrices_type.append(True)

    if 'proU' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'pro'
        get_archive(dir_mat+'ProEM-*use.tar.gz')
        filename = glob.glob(dir_mat+'ProEM-*use.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX4 = np.array(data)
        DXX4 = 1 - DXX4; DXX4 = DXX4 - np.diag(np.diag(DXX4))
        matrices.append(DXX4)
        matrices_type.append(True)

    if 'netB' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'net'
        get_archive(dir_mat+'FiltEM-*bert.tar.gz')
        filename = glob.glob(dir_mat+'FiltEM-*bert.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX5 = np.array(data)
        DXX5 = 1 - DXX5; DXX5 = DXX5 - np.diag(np.diag(DXX5))
        matrices.append(DXX5)
        matrices_type.append(True)

    if 'netU' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'net'
        get_archive(dir_mat+'FiltEM-*use.tar.gz')
        filename = glob.glob(dir_mat+'FiltEM-*use.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX6 = np.array(data)
        DXX6 = 1 - DXX6; DXX6 = DXX6 - np.diag(np.diag(DXX6))
        matrices.append(DXX6)
        matrices_type.append(True)

    print(matrices_type, inData, clustering_algo)
    if os.path.isfile(os.path.join('json', language, clustering_algo+'_'+'_'.join(inData)+'.json')):
        json_file = os.path.join('json', language, clustering_algo+'_'+'_'.join(inData)+'.json')
    else :
        get_archive(dir_mat+'FiltEM-*use.tar.gz')
        filename = glob.glob(dir_mat+'FiltEM-*use.csv')[0]
        df = pd.read_csv(filename, index_col=False, header=None)
        G = nx.from_numpy_matrix(np.array(df))
        partition =  get_refined_partition(matrices, matrices_type, clustering_algo, view_type, k_clusters)
        tw_db['cluster_pred'] = partition
        tw_db.to_csv(os.path.join('labels', lang, clustering_algo+'_'+'_'.join(inData)+'.csv'))
        json_file = get_network(G, tw_db, partition, language, clustering_algo+'_'+'_'.join(inData))
    
    files_in_directory = os.listdir(dir_mat)
    filtered_files = [file for file in files_in_directory if file.endswith(".csv")]
    return json_file


def get_refined_partition(indata, ismatrix, clustering_algo, view='im', k_clusters=16):

    from sklearn.cluster import SpectralClustering
    from sklearn_extra.cluster import KMedoids
    from sklearn.cluster import KMeans
    from multiview.mvsc import MVSC

    for i in range(1,1000):
        if clustering_algo == 'kmeans':
            if view == 'im' or view == 'vad' : partition = KMeans(n_clusters=k_clusters).fit(indata[0]) #IM
            if view == 'pro' or view == 'net': partition = KMedoids(n_clusters=k_clusters, init='random', metric='precomputed').fit(indata[0]) #projection #init=’random’
            partition = partition.labels_
        if clustering_algo == 'sc':
            if view == 'im' or view == 'vad' : partition = SpectralClustering(n_clusters=k_clusters, assign_labels='discretize', affinity='nearest_neighbors').fit(indata[0])
            if view == 'pro' or view == 'net': partition = SpectralClustering(n_clusters=k_clusters, assign_labels='discretize', affinity='precomputed').fit(indata[0])
            partition = partition.labels_
        if clustering_algo == 'mvsc':
            mvsc = MVSC(k=k_clusters)
            clust = mvsc.fit_transform(indata, ismatrix)
            clustering = clust[0]
            clustering = clustering + 1 # mvsc return 0 as the first clusters
            partition = clustering.tolist()
        groups = dict(Counter(list(partition)))
        A= np.array(list(groups.values()))
        if (A>3).all():
            return list(partition)
        break
    print('done!')
    return list(partition)

def get_network(G, data, partitions, lang, name):
    import json

    # Generate the JSON file
    outdata = {}

    outdata['nodes'] = []

    outdata['links'] = []

    for idx, tw in data.iterrows():
        outdata['nodes'].append({
            'id': idx, 'tweet': tw.tweet, 'group': int(partitions[idx])
            })

    for u, v in G.edges():
        outdata['links'].append({
            'source': u, 'target': v, 'value': G[u][v]["weight"]
            })

    with open(os.path.join('json', lang, name+'.json'), 'w') as outfile:
        json.dump(outdata, outfile)

    print("\nThe JSON file with the Twitter network was created...")
    return os.path.join('json', lang, name+'.json')

