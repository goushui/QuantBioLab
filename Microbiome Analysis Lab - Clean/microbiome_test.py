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
  best_score = float('-inf')
  best_match = None
  count = 0
  for s, seq in library.items():
      count+=1
      # print("Checking alignment against sequence ID:", count)
      score = alignment.fast_local_align(sequence, seq, score=alignment.ScoreParam(10, -5, -7), print_output = False)
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
  best_matches = {}
  for query_id, query_kmer in query_kmers.items():
    best_kmer_score, best_kmer_match = KmerMatch(query_kmer, db_kmers)
    best_matches[query_id] = best_kmer_match
  return best_matches

def compute_alignment_matches(db_data, query_data):
  db_alignments = {}
  count =0 
  for query_id, query_seq in query_data.items():
    print(count)
    count += 1
    best_alignment_score, best_alignment = AlignmentMatch(query_seq, db_data)
    db_alignments[query_id] = best_alignment
  return db_alignments

def mutate_sequence(sequence, mutation_rate=0.01, max_length=250):
   mutated_sequence = ""
   for base in sequence[:max_length]:
     if random.random() < mutation_rate:
       mutated_sequence += random.choice("ACGT")
     else:
       mutated_sequence += base
   return mutated_sequence

def mutate_library(library, mutation_rate=0.01, max_length=250):
   mutated_library = {}
   for seq_id, seq in library.items():
     mutated_seq = mutate_sequence(seq, mutation_rate=mutation_rate, max_length=max_length)
     mutated_library[seq_id] = mutated_seq
   return mutated_library
    
def compute_kmer_agreement_list(db_data, query_data, kmer_sizes = [1,3,5,7,9,11,13,15,17,19], local_align_cache_file = "alignment_matches.npy"):
  kmer_agreement = []
  # store alignment matches and load if stored previously to save time
  if os.path.exists(local_align_cache_file):
    alignment_matches = numpy.load(local_align_cache_file, allow_pickle=True).item()
  else:
    alignment_matches = compute_alignment_matches(db_data, query_data)
    numpy.save(local_align_cache_file, alignment_matches)
  for kmer_size in kmer_sizes:
    print(kmer_size)
    kmer_matches = compute_kmer_matches(db_data, query_data, K=kmer_size)
    match_scores = []
    for query_id in query_data.keys():
      if kmer_matches[query_id] == alignment_matches[query_id]:
        match_scores.append(1.0)
      else:
        match_scores.append(0.0)
    kmer_agreement.append(sum(match_scores) / len(match_scores))
  return kmer_agreement

def plot_kmer_agreement(db_data, query_data, filename, local_align_cache_file = "alignment_matches.npy"):
  kmer_sizes = [1,3,5,7,9,11,13,15,17,19]
  kmer_agreement = compute_kmer_agreement_list(db_data, query_data, kmer_sizes=kmer_sizes, local_align_cache_file=local_align_cache_file)
  _ = plt.bar(kmer_sizes, kmer_agreement)
  plt.xlabel("KMer Size")
  plt.ylabel("Average Score of Best")
  plt.title("KMer Agreement Across Different KMer Sizes" + filename)
  plt.xticks(kmer_sizes)
  os.makedirs("results", exist_ok=True)
  plt.savefig(os.path.join("results", filename), dpi=300)
  plt.close()
  
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
  print(f"Generating random split with database size {db_size}, query size {query_size}...")
  db_data, query_data = split_dataset(sequences_16s, db_size=db_size, query_size=query_size)
  print("Generating mutated libraries...")
  illumina_db = mutate_library(db_data, mutation_rate=0.01, max_length=250)
  illumina_query = mutate_library(query_data, mutation_rate=0.01, max_length=250)
  nanopore_db = mutate_library(db_data, mutation_rate=0.1, max_length=10000)
  nanopore_query = mutate_library(query_data, mutation_rate=0.1, max_length=10000)
  print("Computing kmer agreement...")
  plot_kmer_agreement(db_data, query_data, filename="full.png", local_align_cache_file="full_alignment_matches.npy")
  plot_kmer_agreement(illumina_db, illumina_query, filename="illumina.png", local_align_cache_file="illumina_alignment_matches.npy")
  plot_kmer_agreement(nanopore_db, nanopore_query, filename="nanopore.png", local_align_cache_file="nanopore_alignment_matches.npy")
  
  
     
   
   
