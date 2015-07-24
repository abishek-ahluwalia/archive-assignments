library(tm)
library(SnowballC)
library(stringdist)
# Create corpus
createCorpus <-function(names0){ 

#Remove Company stopWords
output = findCompanyStopWords(corpus0, stopFreqThreshold = thresholdFrequencyStopWords )
companyStopWords = output$stop
freq = output$freq
corpus0 = tm_map(corpus0, removeWords,  companyStopWords )

#convert Corpus
corpus0P = convertCorpus(corpus0)


#sort each record 

corpus0P = sortCorpus(corpus0P)
corpus0P = str_trim(corpus0P )


#createDistances
#distances <- as.dist(stringdistmatrix(newCorpus , newCorpus))

output = list("corpus" = corpus0P , "stop" = companyStopWords , 'freq' = freq)
output
}
