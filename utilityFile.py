import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

import nltk
#from models import InferSent
#import torch
from textblob import TextBlob
from gensim.models import doc2vec
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
#from sentence_transformers import SentenceTransformer,models,SentencesDataset

#model_sent = SentenceTransformer('bert-base-nli-mean-tokens')

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    docs=[item for sublist in documents for item in sublist]
    docs=list(set(docs))
    idfDict = dict.fromkeys(docs, 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
            else:
                print(word,item)

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def embedSentences(sentences):
    sentence_embeddings = model_sent.encode(sentences)
    return sentence_embeddings

def scenesSentenceEmbedding(sceneTexts):
    sceneEmbedings=[]
    for i in range(len(sceneTexts)):
        print(i)
        if sceneTranscript[i]!='':
            txtblb=TextBlob(str(sceneTexts[i].lower()))
            sent=[str(j) for j in txtblb.sentences]
            sceneSentEmd=embedSentences(sent)
            sceneEmbedings.append(sceneSentEmd)
    np.save('SentenceEmbeddings_BERT_XLNet',np.array(sceneEmbedings))
    return np.array(sceneEmbedings)

def doc2vecEmbedding(sceneTexts):
    model= Doc2Vec.load("d2v.model")
    #Document Embedding data
    doc2vec_Emb=[]
    for i in range(len(sceneTexts)):
        doc2vec_Emb.append( model.infer_vector(sceneTexts[i].lower()))
    doc2vec_Emb=np.array(doc2vec_Emb)
    np.save('Scene_Doc2Vec_Embedding',np.array(doc2vec_Emb))
    return doc2vec_Emb

def list_oneHot_encode(scene_listName):
    allNames=[item for sublist in scene_listName for item in sublist]
    uniqueNames=list(set(allNames))
    names = [name for name in uniqueNames if '#' not in name]
    data = {v: k for k, v in enumerate(names)}
    existingName_rep=[]
    for sln in scene_listName:
        sceneName_rep=np.zeros(len(uniqueNames))
        for n in sln:
            if n in data:
                key=data[n]
                sceneName_rep[key]=1
        existingName_rep.append(sceneName_rep)
    existingName_Data=np.array(existingName_rep)
    return existingName_Data, data

def getTFIDF_Representation(sceneData):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sceneData)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    sceneTFIDF_Array=np.array(denselist)
    return sceneTFIDF_Array
