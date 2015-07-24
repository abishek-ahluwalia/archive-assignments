library(tm)
library(SnowballC)
library(stringdist)



# Create corpus data preprocessing
dataPreprocessing<- function(names0 ){ 
  
  corpus0 = Corpus(VectorSource(names0))#[start:end]))
  
  # Convert to lower-case
  corpus0 = tm_map(corpus0, tolower)
  
  corpus0 = tm_map(corpus0, PlainTextDocument)
  
  # Replace punctuation with whitespace
  removePunctuation <- function(x) UseMethod("removePunctuation", x)
  removePunctuation.PlainTextDocument <- function(x) gsub("[[:punct:]]+",
                                                          " ", x)
  corpus0 = tm_map(corpus0, removePunctuation)
  
  corpus0 = tm_map(corpus0, PlainTextDocument)
  
  
  # Remove stopwords 
  
  
  corpus0 = tm_map(corpus0, removeWords,  stopwords("english"))
  
  
  # Stem document 
  
  corpus0= tm_map(corpus0, stemDocument)
  
#createDistances moved to new file
#distances <- as.dist(stringdistmatrix(newCorpus , newCorpus))
#distances = findDistancesFromCorpus(corpus0P , method = "lv")
#distances = as.dist(distances)


 corpus0
}


