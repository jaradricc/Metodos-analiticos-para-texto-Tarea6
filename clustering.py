# -*- coding: utf-8 -*-
import numpy as np
import gensim
from scipy.spatial import distance

def distancia_euclidiana(a,b):
    return distance.euclidean(a,b)

def distancia_hausdorf(a,b):
    h = 0.0
    for ia in a:
        d_min = float('Inf')
        for ib in b:
            aux = distancia_euclidiana(ia,ib)
            if aux < d_min:
                d_min = aux
        if d_min > h:
            h = d_min
    return h


def k_medias_euclidiano(frases , k):
    centroides = map(lambda i: frases[i], np.random.choice(len(frases), k) ) # Una lista con los k centroides
    clusters = np.random.choice(len(centroides), len(frases)) # Una lista con el índice del centroide asignado a cada frase
    repeat = True
    while repeat:
        repeat = False
        for ip in range(len(frases)):
            min_d = distancia_euclidiana(frases[ip], centroides[clusters[ip]]) # la distancia entre la frase y el centroide que tiene asignado.
            for ic in range(len(centroides)):
                if ic != clusters[ip]:
                    aux = distancia_euclidiana(frases[ip], centroides[ic])
                    if aux < min_d:
                        min_d = aux
                        clusters[ip] = ic
                        repeat = True
        if repeat: # Si se necesitan redefinir los centroides
            for i in range(k):
                if sum(clusters == i) > 0 :
                    centroides[i] = sum(frases[clusters == i]) / sum(clusters == i)
    return clusters


arch = open('frases.txt', 'r')
corpus = list()

for l in arch:
    if l != "" :
        corpus.append(l.split("\t"))
arch.close()
corpus = dict(corpus)

## Preparamos las oraciones para meterlas a word2Vec que las recibe a manera de listas de listas de palabras
phrases = list(map(lambda l: l.split(),corpus.keys() ))
model = gensim.models.Word2Vec(phrases,min_count=1)
##

##la composición por suma queda como sigue:
comp_sum = map(lambda p: sum(map(lambda w: model[w], p)), phrases)

##la compisición por multiplicación de elemento por elemento queda como sique:
comp_mult = list()
for p in phrases:
    aux = np.ones(len(model[p[0]]))
    for w in p:
        aux *= model[w]
    comp_mult.append(aux)


## Composición como matriz
matrix_phrases = map(lambda p: map(lambda w: model[w], p), phrases)

## Ejecución de k-medias usando la composicion de la suma y distancia euclidiana

clusters_suma = k_medias_euclidiano(np.array(comp_sum), len(set(corpus.values())))

clusters_mult = k_medias_euclidiano(np.array(comp_mult), len(set(corpus.values())))

# clusters_haus = k_medias_haus(np.array(matrix_phrases), len(set(corpus.values())))
