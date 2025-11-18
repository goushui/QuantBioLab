# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:32:48 2018

@author: jdkan
"""
import numpy
import time
import alignment
import copy
import random
import matplotlib




def Load16SFastA(path, fraction = 1.0):
    # from a file, read in all sequences and store them in a dictionary
    # sequences can be randomly ignored for testing by adjusting the fraction
    random.seed(11)
    
    infile = open(path, 'r')
    sequences_16s = {}
    c = 0
    my_seq = ""
    for line in infile:
        if ">" in line:
            my_id = line[1:-1]
            if random.random() < fraction:
                sequences_16s[my_id] = ""
            
            
        else:
            if my_id in sequences_16s:
                sequences_16s[my_id] += line[:-1]
    
       
    return sequences_16s



def ConvertLibaryToKmerSets(library, K=2):
    
    new_lib = {}
    c = 0
    for k in library.keys():
        new_lib[k] = set()
        if len(library[k]) < K: continue
        for i in range(len(library[k])-K+1):
            new_lib[k].add(library[k][i:i+K])
        # add your code here to build the k-mer set
        
    return new_lib

def JaccardIndex(s1, s2):
    numerator = float(len(s1.intersection(s2)))
    denominator = float(len(s1.union(s2)))
    return numerator/denominator

def KmerMatch(sequence_kmer_set, library_kmer_set):
    best_score = 0.0
    best_match = None
    
    for s, s_kmers in library_kmer_set.items():
      score = JaccardIndex(s_kmers, sequence_kmer_set)
      if score > best_score:
          best_score = score
          best_match = s
    return best_score, best_match


def AlignmentMatch(sequence, library):
    best_score = -10000000000
    best_match = None
    
    for s, seq in library.items():
        score = alignment.local_align(sequence, seq, score=alignment.ScoreParam(10, -5, -7), print_output = False)
        if score[0] > best_score:
            best_score = score[0]
            best_match = s
    return best_score, best_match

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   

  
   fn = "bacterial_16s_genes.fa"
   sequences_16s = Load16SFastA(fn, fraction = 1.0)
   
   
   print ("Loaded %d 16s sequences." % len(sequences_16s))
   
   
   kmer_16s_sequences = ConvertLibaryToKmerSets(sequences_16s, K=6)
   
   
