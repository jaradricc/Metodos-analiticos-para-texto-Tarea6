# -*- coding: utf-8 -*-
import numpy as np
import gensim
import math
from scipy.spatial import distance
from itertools import combinations
from random import shuffle

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

def distancia_frobenius(a,b):
    a = np.matrix(a)
    b = np.matrix(b)
    aux = a * b.T
    d = np.trace(aux)
    d = math.sqrt(math.fabs(d))
    return d

def calcula_distancias(frases, alg):
    distancias = np.zeros((len(frases), len(frases)))
    for x, y in combinations(range(len(frases)), 2):
        if alg == 0:
            d = distancia_hausdorf(frases[x], frases[y])
        else:
            d = distancia_frobenius(frases[x], frases[y])
        distancias[x,y] = d
        distancias[y,x] = d
    return distancias


def major_clust(frases, alg):  # el parámetro alg define el algoritmo de distancia a utilizar: 1 = hausdorf y cualquier otro = frobenius
    clusters = np.array(range(1,len(frases) + 1 )) # cada nodo es un clúster
    # print 'clusters: ', clusters
    ## calculamos las distancias hausdorf entre todos los nodos.
    distancias = calcula_distancias(frases,alg)
    # print "distancias: ", len(distancias), len(distancias[0]), '\n',distancias
    ##
    t = False
    while not t:
        t = True
        for u in range(len(frases)):
            acum = np.zeros(len(frases))
            for i in range(len(frases)):
                acum[clusters[i]-1] += distancias[u,i]
            minimo = float("inf")
            for w in range(len(frases)):
                if acum[w] < minimo and acum[w] != 0:
                    minimo = acum[w]
                    wmin = w
            if clusters[u] != (wmin + 1):
                clusters[u] = wmin + 1
                t = False

    return clusters


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


def validation(keys, other):
    ## un diccionario con las etiquetas de los clusters y una lista de las posiciones en el arreglo que pertenecen a dicho cluster.
    clustersk = dict()
    for i in range(len(keys)):
        if clustersk.has_key(keys[i]):
            clustersk[keys[i]].append(i)
        else:
            clustersk[keys[i]] = [i]
    ###
    ## probamos cada uno de los clusters y le asignamos la etiqueta del que mas se parece
    ## llenamos el diccionario con los clusters generados.
    clusters = dict()
    for i in range(len(other)):
        if clusters.has_key(other[i]):
            clusters[other[i]].append(i)
        else:
            clusters[other[i]] = [i]
    ##recorremos los clusters comparándolos contra las etiquetas para asignarlas al clúster que mas se parecen
    resultado = list()
    for i in clusters.items():
        sim = 0
        for j in clustersk.items():
            aux = 0
            ## contamos cuantos elementos coinciden con el clusterk
            for x in i[1]:
                if x in j[1]:
                    aux += 1
            ## si se parece más al cluster, decimos que cluster corresponde al clustersk
            if aux >= sim:
                sim = aux
                tag = str(j[0])
        resultado.append([i[0], sim, tag]) #(cluster, items iguales, tag)
    return resultado

arch = open('frases.txt', 'r')
corpus = list()
keys = list()

for l in arch:
    if l != "" :
        aux = l.split("\t")
        corpus.append(aux[0])
        keys.append(aux[1].rstrip())
arch.close()

## Preparamos las oraciones para meterlas a word2Vec que las recibe a manera de listas de listas de palabras
phrases = list(map(lambda l: l.split(),corpus ))
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

clusters_suma = k_medias_euclidiano(np.array(comp_sum), len(corpus))

clusters_mult = k_medias_euclidiano(np.array(comp_mult), len(corpus))

clusters_haus = major_clust(np.array(matrix_phrases), 1)

clusters_frob = major_clust(np.array(matrix_phrases),0)

## Evaluamos resultados
resultados_suma = validation(keys, clusters_suma)
print 'correctamente clasificados con la composición de la suma: ', sum(map(lambda i: i[1],resultados_suma))
resultados_mult = validation(keys, clusters_mult)
print 'correctamente clasificados con la composición de la multiplicación: ', sum(map(lambda i: i[1],resultados_mult))
resultados_haus = validation(keys, clusters_haus)
print 'correctamente clasificados con la distancia Hausdorf: ', sum(map(lambda i: i[1],resultados_haus))
resultados_frob = validation(keys, clusters_frob)
print 'correctamente clasificados con la distancia Frobenius: ', sum(map(lambda i: i[1],resultados_frob))
