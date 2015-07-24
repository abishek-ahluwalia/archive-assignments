chunkSize = 130000

# enter the index for algorithms to be used in clustering 
#and duplicate detection algos below from the list
# 1 direct Key matching 
# 2 fingerprint matching
# 3 n gram finger print matching (not implemented but api available in jar attached)  
# 4 Phonetic Fingerprint ( soundex)
# 5 Phonetic Fingerprint ( metaphone ) ( not implemented refer to PGRdup R package)
# 6 Levenshtein Distance (edit distance)
# 7 PPM (not implemented but api available in jar attached) 
# 8 (not implemented) OPTICS algorithm . 
# 9 Nearest Neighbor Methods (not implemented but api available in jar attached)
clusteringAlgoParameters = c(1    )

deduplicationAlgoParamters = c(1   )

thresholdFrequencyStopWords = 120

# maximum distance between strings to be in same cluster
editDistanceThreshold = 3
