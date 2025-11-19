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
import matplotlib.pyplot as plt
import os




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

def split_dataset(data, db_size=200, query_size=50):
    data = list(data.items())
    random.shuffle(data)
    db_data = data[:db_size]
    query_data = data[db_size:db_size + query_size]
    db_data = {k: v for k, v in db_data}
    query_data = {k: v for k, v in query_data}
    return db_data, query_data

def compute_kmer_matches(db_data, query_data, K=2):
  db_kmers = ConvertLibaryToKmerSets(db_data, K=K)
  query_kmers = ConvertLibaryToKmerSets(query_data, K=K)
  best_scores = {}
  for query_id, query_kmer in query_kmers.items():
    best_kmer_score, _ = KmerMatch(query_kmer, db_kmers)
    best_scores[query_id] = best_kmer_score
  return best_scores

def compute_alignment_matches(db_data, query_data):
  db_alignments = {}
  for query_id, query_seq in query_data.items():
    best_alignment_score, _ = AlignmentMatch(query_seq, db_data)
    db_alignments[query_id] = best_alignment_score
  return db_alignments

    
    
  
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   

  
  fn = "bacterial_16s_genes.fa"
  sequences_16s = Load16SFastA(fn, fraction = 1.0)
  
  
  print ("Loaded %d 16s sequences." % len(sequences_16s))
  
  
  # kmer_16s_sequences = ConvertLibaryToKmerSets(sequences_16s, K=2)
  # kmer_16s_sequences = {k: v for k, v in list(kmer_16s_sequences.items())[:5]}
  # base_id, base_seq = list(kmer_16s_sequences.items())[0]
  # print ("First 5 sequences:")
  # for k, v in kmer_16s_sequences.items():
  #   print(f"{k}: {v}")
  # best_score, best_match = KmerMatch(base_seq, kmer_16s_sequences)
  # print ("Best match score:", best_score)
  # print ("Best match ID:", best_match)
  # alignment_score, alignment_match = AlignmentMatch(base_seq, sequences_16s)
  # print ("Best alignment score:", alignment_score)
  # print ("Best alignment ID:", alignment_match)
  
  
  db_size = 200
  query_size = 50
  random.seed(42)
  db_data, query_data = split_dataset(sequences_16s, db_size=db_size, query_size=query_size)
  kmer_sizes = [1,3,5,7,9,11,13,15,17,19]
  kmer_agreement = []
  for kmer_size in kmer_sizes:
     kmer_scores = compute_kmer_matches(db_data, query_data, K=kmer_size)
     kmer_agreement.append(sum(kmer_scores.values()) / len(kmer_scores))
  _ = plt.bar(kmer_sizes, kmer_agreement)
  plt.xlabel("KMer Size")
  plt.ylabel("Average Score of Best")
  plt.title("KMer Agreement Across Different KMer Sizes")
  plt.xticks(kmer_sizes)
  os.makedirs("results", exist_ok=True)
  
  plt.savefig("results/kmer_agreement.png")
  # plt.show()
  
     
   
   
