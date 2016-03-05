import numpy as np
import os
from scipy.sparse import lil_matrix
import pickle
datafile = '../data/freebase/train_single_relation.txt'

entities = set()
relations = set()
with open(datafile, 'r') as f:
    for line in f.readlines():
        start, relation, end = line.split('\t')
        if start.strip() not in entities:
            entities.add(start.strip())
        if end.strip() not in entities:
            entities.add(end.strip())
        if relation.strip() not in relations:
            relations.add(relation)


n_entities = len(entities)
entities = list(entities)
entity_dic = {entities[k]:k for k in range(len(entities))}

n_relations = len(relations)
relations = list(relations)
relation_dic = {relations[k]:k for k in range(len(relations))}

selected_relations = list() #manually selected list of relations
selected_relations.append(relation_dic['place_of_birth'])
selected_relations.append(relation_dic['place_of_death'])
selected_relations.append(relation_dic['nationality'])
selected_relations.append(relation_dic['location'])

entity_count = np.zeros(n_entities)
T = [lil_matrix((n_entities, n_entities), dtype=int) for k in range(n_relations)]
cnt = 0
with open(datafile, 'r') as f:
    for line in f.readlines():
        start, relation, end = line.split('\t')
        e_i = entity_dic[start.strip()]
        e_j = entity_dic[end.strip()]
        r_k = relation_dic[relation.strip()]
        T[r_k][e_i,e_j] = 1
        if r_k in selected_relations:
            if e_i == e_j:
                entity_count[e_i] += 1
            else:
                entity_count[e_i] += 1
                entity_count[e_j] += 1
T = [X.tocsr() for X in T]
entities = np.array(entities)
relations = np.array(relations)

n_list = [2000,3000,4000,5000]
for n_new_entity in n_list:
    new_idx = sorted(range(len(entity_count)), key=lambda i: entity_count[i], reverse=True)[:n_new_entity]

    newT = np.zeros([len(selected_relations), n_new_entity, n_new_entity])
    for idx, k in enumerate(selected_relations):
        newT[idx] = T[k][new_idx, :][:, new_idx].todense()

    print(n_new_entity)
    print('num triple', np.sum([X.sum() for X in newT]))
    print('sparsity', np.sum([X.sum() for X in newT]) / (n_new_entity**2 * n_relations))
    pickle.dump([newT, entities[new_idx], relations[selected_relations]], open('../data/freebase/subset_%d.pkl'%(n_new_entity),'wb'))

