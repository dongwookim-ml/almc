import numpy as np
import pickle
import os
from operator import itemgetter

dest = '../data/wordnet/'
input_tensor = 'wordnet_csr.pkl'
output_tensor = 'reduced_wordnet.pkl'

T = pickle.load(open(os.path.join(dest, input_tensor), 'rb'))

n_relation = len(T)
n_entity, _ = T[0].shape

print(n_entity)
print(n_relation)
a = np.sum([T[k].nnz for k in range(n_relation)])
b = n_relation * n_entity ** 2
print(a, b, a / b)

degree = [0 for i in range(n_entity)]
for i in range(n_entity):
    degree[i] = np.sum([T[k].getcol(i).nnz for k in range(n_relation)])
    degree[i] += np.sum([T[k].getrow(i).nnz for k in range(n_relation)])

sorted_entity = sorted(enumerate(degree), key=itemgetter(1), reverse=True)

n_reduce_entity = 1000
reduced_entity = [k[0] for k in sorted_entity[:n_reduce_entity]]

newT = np.zeros([n_relation, n_reduce_entity, n_reduce_entity])
for k in range(n_relation):
    for i, ei in enumerate(reduced_entity):
        for j, ej in enumerate(reduced_entity):
            newT[k, i, j] = T[k][ei, ej]

print(np.sum(newT))
valid_relations = list()
for k in range(n_relation):
    k_sum = np.sum(newT[k])
    print('relation', k, np.sum(newT[k]))
    if k_sum != 0:
        valid_relations.append(k)

valid_entity = list()
for i in range(n_reduce_entity):
    e_sum = np.sum(newT[:, i, :]) + np.sum(newT[:, :, i])
    if e_sum == 0:
        print(i)
    else:
        valid_entity.append(i)

realT = np.zeros((len(valid_relations), len(valid_entity), len(valid_entity)))
for k, oldk in enumerate(valid_relations):
    for i, ei in enumerate(valid_entity):
        for j, ej in enumerate(valid_entity):
            realT[k, i, j] = newT[oldk, ei, ej]

pickle.dump(realT, open(os.path.join(dest, output_tensor), 'wb'))
print(len(valid_entity), len(valid_relations))
