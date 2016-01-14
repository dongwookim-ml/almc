import numpy as np
import os
import pickle
from scipy.sparse import *

dest = '../data/wordnet/'
entity_file = 'entities.txt'
relation_file = 'relations.txt'
tensor_files = ['train.txt']
output_tensor = 'wordnet_csr.pkl'

entity_dict = {line.strip(): i for (i, line) in enumerate(open(os.path.join(dest, entity_file), 'r').readlines())}
relation_dict =  {line.strip(): i for (i, line) in enumerate(open(os.path.join(dest, relation_file), 'r').readlines())}

n_relations = len(relation_dict)
n_entities = len(entity_dict)

T = [csr_matrix((n_entities,n_entities), dtype=int) for i in range(n_relations)]

for tensor_file in tensor_files:
    lines = open(os.path.join(dest, tensor_file), 'r').readlines()
    for line_no, line in enumerate(lines):
        entity_i, relation, entity_j = line.split('\t')
        i = entity_dict[entity_i.strip()]
        j = entity_dict[entity_j.strip()]
        k = relation_dict[relation.strip()]
        T[k][i, j] = 1
        if line_no % 1000 == 0:
            print(line_no)

pickle.dump(T, open(os.path.join(dest, output_tensor), 'wb'))
