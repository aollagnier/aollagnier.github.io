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

@app.route('/api/characterization/<string:group>/<string:ngrams_method>')
def charaterization(group, ngrams_method):
    from collections import Counter
    views = config.split('-')
    clustering_algo = views[0]

    if '_' in views[1]: inData = views[1]
    else:   inData = views[1].split('_')
    print('***', ngrams_method)

    group = int(group)
    #if '_' in views[1]: df = pd.read_csv(os.path.join('labels', language, clustering_algo+'_'+''.join(inData)+'.csv'), index_col=False)
    #else: 
        
    df = pd.read_csv(os.path.join('labels', language, clustering_algo+'_'+''.join(inData)+'.csv'), index_col=False)

    grouped_df =  df[df['cluster_pred']==group]
    
    head = list((i for (i, v) in Counter(grouped_df['label_mix']).most_common()))
    ngrams_top, ngrams_full = get_ngrams(grouped_df, df, method=ngrams_method)

    words_viz = word_visualization(ngrams_full, group, os.path.join('labels', language, clustering_algo+'_'+''.join(inData)+'_'+str(group)+'.json'))
    #words_viz = word_visualizationV2(ngrams_full, group, os.path.join('labels', language, clustering_algo+'_'+''.join(inData)+'_'+str(group)+'.json'))
    
    return {'head':head[0], 'ngrams':ngrams_top, 'group':group, 'json':words_viz}


# def word_visualizationV2(ngrams, file_name):
    from collections import Counter
    import networkx as nx
    import statistics
    import json
    
    bigrams = list((tuple(i.split()) for i in ngrams if len(i.split()) == 2))
    trigrams = list((tuple(i.split()) for i in ngrams if len(i.split()) == 3))

    bigrams = list(((*num, 2) for item in bigrams for num in (item if isinstance(item, list) else (item,))))
    trigrams = list(((*num[::len(num)-1],1) for item in trigrams for num in (item if isinstance(item, list) else (item,))))

    df_result = pd.DataFrame(bigrams+trigrams, columns =['word 1', 'word 2', 'weight'])
    words = list(set(df_result['word 1'].tolist() + df_result['word 2'].tolist()))
    w_dict = Counter(df_result['word 1'].tolist() + df_result['word 2'].tolist())
    #print(len(w_dict), w_dict)
    threshold= statistics.mean(w_dict.values())
    #print('Number of words: {}, Avg Frequency: {}'.format(len(words), statistics.mean(w_dict.values())))
    aof = list((i for i in words if w_dict[i] < threshold))
    #print(len(aof))
    G = nx.from_pandas_edgelist(df_result,'word 1', 'word 2', edge_attr='weight')
    G.remove_nodes_from(aof)
    #print("\nCreation the Tweet Graph: {} Nodes, {} Edges".format(len(G.nodes()), len(G.edges())))

    # Create the basic structure of a force-directed D3 network
    d3_graph = {
        'nodes': [],
        'links': [],
    }

    # Add each node in the KKG network to the D3 network
    for node in G.nodes():
        d3_graph_node = {
            'id': words.index(str(node)),
            'tweet': str(node),
            'group':group
        }
        
        #for centrality_measure in centrality_measures:
            #d3_graph_node['centrality'][centrality_measure] = centrality_measures[centrality_measure][node]
        d3_graph['nodes'].append(d3_graph_node)

    # Add each edge in the KKG network to the D3 network
    for u, v in G.edges():
            d3_graph['links'].append({
                    'source': words.index(u), 'target': words.index(v), 'value': G[u][v]['weight']
                    })

    # Output the network graph in JSON format
    with open(file_name, 'w') as f:
        json.dump(d3_graph, f, indent=2)
    return file_name

def word_visualization(ngrams, group, file_name):
    from collections import Counter
    import networkx as nx
    import statistics
    import json
    
    bigrams = list((tuple(i.split()) for i in ngrams if len(i.split()) == 2))
    trigrams = list((tuple(i.split()) for i in ngrams if len(i.split()) == 3))

    bigrams = list(((*num, 2) for item in bigrams for num in (item if isinstance(item, list) else (item,))))
    trigrams = list(((*num[::len(num)-1],1) for item in trigrams for num in (item if isinstance(item, list) else (item,))))

    df_result = pd.DataFrame(bigrams+trigrams, columns =['word 1', 'word 2', 'weight'])
    words = list(set(df_result['word 1'].tolist() + df_result['word 2'].tolist()))
    w_dict = Counter(df_result['word 1'].tolist() + df_result['word 2'].tolist())
    #print(len(w_dict), w_dict)
    threshold= statistics.mean(w_dict.values())
    #print('Number of words: {}, Avg Frequency: {}'.format(len(words), statistics.mean(w_dict.values())))
    aof = list((i for i in words if w_dict[i] < threshold))
    #print(len(aof))
    G = nx.from_pandas_edgelist(df_result,'word 1', 'word 2', edge_attr='weight')
    G.remove_nodes_from(aof)
    #print("\nCreation the Tweet Graph: {} Nodes, {} Edges".format(len(G.nodes()), len(G.edges())))

    # Create the basic structure of a force-directed D3 network
    d3_graph = {
        'nodes': [],
        'links': [],
    }

    # Add each node in the KKG network to the D3 network
    for node in G.nodes():
        d3_graph_node = {
            'id': words.index(str(node)),
            'tweet': str(node),
            'group':group
        }
        
        #for centrality_measure in centrality_measures:
            #d3_graph_node['centrality'][centrality_measure] = centrality_measures[centrality_measure][node]
        d3_graph['nodes'].append(d3_graph_node)

    # Add each edge in the KKG network to the D3 network
    for u, v in G.edges():
            d3_graph['links'].append({
                    'source': words.index(u), 'target': words.index(v), 'value': G[u][v]['weight']
                    })

    # Output the network graph in JSON format
    with open(file_name, 'w') as f:
        json.dump(d3_graph, f, indent=2)
    return file_name


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

def get_ngrams(df_group, df_global, threshold = 0.001, context_window = 2, top_n=10, method = 'freq'):
    from nltk.util import ngrams
    from nltk.util import everygrams
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    # ngrams_list_o = list((list(everygrams(i.split(), min_len=2, max_len=3)) for i in df.clean_tw.values))
    # ngrams_list = [item for sublist in ngrams_list_o for item in sublist]

    # frequencies = most_common(ngrams_list)
    # percentages = [(instance, count / len(ngrams_list)) for instance, count in frequencies]
    # filtered_bgrams = [instance for instance, count in percentages if count >= threshold]
    
    # ngrams = string_set(sorting(filtered_bgrams))
    # return list((' '.join(k) for k, v in dict(frequencies).items() if k in ngrams))

    
    if method == 'pearson':
        from collections import defaultdict
        from scipy import stats

        df_global['clean_tw'] = df_global['clean_tw'].apply(lambda x: cleaning(x))
        vectorizer = TfidfVectorizer(ngram_range=(2, 3))

        corpus =[]
        
        for name, grouped_df in df_global.groupby(['cluster_pred']):
            corpus.append(' '.join(grouped_df['clean_tw'].values))
            
        tfs = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        dense = tfs.todense()
        denselist = dense.tolist()
        df_result_global = pd.DataFrame(denselist, columns=feature_names)
        
        words_d_global = defaultdict(dict,{ k:{} for k in feature_names })
        
        for k, val in words_d_global.items():
            new_val = df_result_global[k].values
            words_d_global[k] = {key:v for key, v in enumerate(new_val)}


        words_d_local = defaultdict(dict,{ k:{key:0 for key in range(len(df_group['cluster_pred'].unique().tolist()))} for k in feature_names })

        corpus_local = []

        for name, grouped_df in df_group.groupby(['cluster_pred']):
            vectorizer = TfidfVectorizer(ngram_range=(2, 3))
            corpus_local.append(' '.join(grouped_df['clean_tw'].values))
        
        tfs_local = vectorizer.fit_transform(corpus_local)
        feature_names = vectorizer.get_feature_names()
        dense = tfs_local.todense()
        denselist = dense.tolist()
        df_result_local = pd.DataFrame(denselist, columns=feature_names)
        local_tfidf = df_result_local.sum(axis = 0)
        local_tfidf = local_tfidf.values
    
        for idx, w in enumerate(feature_names):
            words_d_local[w][name-1] = local_tfidf[idx]
            
        for name, grouped_df in df_group.groupby(['cluster_pred']):
            attrib_to_degree = {}
            vectorizer = TfidfVectorizer(ngram_range=(2, 3))
            tfs_local = vectorizer.fit_transform(grouped_df['clean_tw'].values)
            feature_names = vectorizer.get_feature_names()
            
            for k, w in words_d_local.items():
                if k in feature_names:
                    local = list(w.values())
                    glob = list(words_d_global[k].values())
                    attrib_to_degree[k] = stats.pearsonr(local, glob)[0]
            ngrams_set = sorted( attrib_to_degree, key=attrib_to_degree.get, reverse=True )[:10]

    if method == 'freq':
        df_group['clean_tw'] = df_group['clean_tw'].apply(lambda x: cleaning(x))

        vectorizer = CountVectorizer(ngram_range=(2, 3))
        #vectorizer = TfidfVectorizer(ngram_range=(2, 3), use_idf=False, min_df = 0.05)
        tfs = vectorizer.fit_transform([' '.join(df_group['clean_tw'].values)])
        sum_words = tfs.sum(axis=0) 

    if method == 'tfidf':
        corpus = []
        vectorizer = TfidfVectorizer(ngram_range=(2, 3))

        for name, grouped_df in df_global.groupby(['cluster_pred']):
            corpus.append(' '.join(grouped_df['clean_tw'].values))

        tfs = vectorizer.fit_transform(corpus)
        group = df_group['cluster_pred'].unique().tolist()
        sum_words = tfs[group[0]].sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngrams_set_full =sorted(words_freq, key = lambda x: x[1], reverse=True)

    ngrams= list((v[0] for v in ngrams_set_full[:top_n]))
    ngrams_set_top = string_set(ngrams)
    idx_inf = top_n
    idx_sup = idx_inf+(top_n-len(ngrams_set_top))

    while len(ngrams_set_top) < top_n:
        ngrams= list((v[0] for v in ngrams_set_full[idx_inf:idx_sup]))
        print('ngrams:', ngrams)
        ngrams= ngrams_set_top+ngrams
        print('ngrams:', ngrams)
        ngrams_set_top = list(set(string_set(ngrams)))
        idx_inf = idx_sup
        idx_sup = idx_inf +(top_n-len(ngrams_set_top))
    ngrams_set_full= list((v[0] for v in ngrams_set_full))
    return ngrams_set_top, ngrams_set_full

def get_archive(file_path, dir_mat):
    import tarfile
    with  tarfile.open(file_path, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, dir_mat)
        tar.close()

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
    print(views)
    clustering_algo = views[0]

    dir_mat = 'matrices/'+language+'/'
    
    matrices = []
    matrices_type = [] #False feature set / True feature set
    print(views[1])
    inData = views[1].split('_')
    view_type = 'im'
    print('-----',views, inData)
    if 'bert' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'im'
        get_archive(glob.glob(dir_mat+'EM-*bert.tar.gz')[0], dir_mat)
        filename = glob.glob(dir_mat+'EM-*bert.csv')[0]
        data = pd.read_csv(filename,index_col=False,header=None)
        DX1 = data.values
        matrices.append(DX1)
        matrices_type.append(False)

    if 'use' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'vad'
        get_archive(glob.glob(dir_mat+'EM-*use.tar.gz')[0], dir_mat)
        filename = glob.glob(dir_mat+'EM-*use.csv')[0]
        data = pd.read_csv(filename, index_col=False,header=None)
        DX2 = data.values
        matrices.append(DX2)
        matrices_type.append(False)

    if 'proB' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'pro'
        get_archive(glob.glob(dir_mat+'ProEM-*bert.tar.gz')[0], dir_mat)
        filename = glob.glob(dir_mat+'ProEM-*bert.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX3 = np.array(data)
        DXX3 = 1 - DXX3; DXX3 = DXX3 - np.diag(np.diag(DXX3))
        matrices.append(DXX3)
        matrices_type.append(True)

    if 'proU' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'pro'
        get_archive(glob.glob(dir_mat+'ProEM-*use.tar.gz')[0], dir_mat)
        filename = glob.glob(dir_mat+'ProEM-*use.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX4 = np.array(data)
        DXX4 = 1 - DXX4; DXX4 = DXX4 - np.diag(np.diag(DXX4))
        matrices.append(DXX4)
        matrices_type.append(True)

    if 'netB' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'net'
        get_archive(glob.glob(dir_mat+'FiltEM-*bert.tar.gz')[0], dir_mat)
        filename = glob.glob(dir_mat+'FiltEM-*bert.csv')[0]
        data = pd.read_csv(filename, index_col=False, header=None)
        DXX5 = np.array(data)
        DXX5 = 1 - DXX5; DXX5 = DXX5 - np.diag(np.diag(DXX5))
        matrices.append(DXX5)
        matrices_type.append(True)

    if 'netU' in inData:
        if clustering_algo == 'kmeans' or clustering_algo == 'sc' : view_type = 'net'
        get_archive(glob.glob(dir_mat+'FiltEM-*use.tar.gz')[0], dir_mat)
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
        get_archive(glob.glob(dir_mat+'FiltEM-*use.tar.gz')[0], dir_mat)
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

