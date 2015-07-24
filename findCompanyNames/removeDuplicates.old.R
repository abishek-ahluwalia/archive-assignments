library(tm)
library(SnowballC)
library(stringdist)
removeDuplicates <- function(corpus){
  
  # Create corpus
  #corpus0 = Corpus(VectorSource(names0[1:5000]))
  
  # Convert to lower-case
  #corpus0 = tm_map(corpus0, tolower)
  
  #corpus0 = tm_map(corpus0, PlainTextDocument)
  
  # Replace punctuation with whitespace
  removePunctuation <- function(x) UseMethod("removePunctuation", x)
  removePunctuation.PlainTextDocument <- function(x) gsub("[[:punct:]]+",
                                                          " ", x)
  #corpus0 = tm_map(corpus0, removePunctuation)
  
  #corpus0 = tm_map(corpus0, PlainTextDocument)
  
  
  # Remove stopwords 
  companyStopW = findCompanyStopWords(corpus0)
  companyStopWords <- function(x) removeWords(x ,companyStopW )
  stopWords <-function(x) removeWords(x , stopWords("english"))
  #corpus0 = tm_map(corpus0, removeWords,  stopwords("english"))
  
  #add company stop words 
  
  
  # Stem document 
  
  #corpus0= tm_map(corpus0, stemDocument)
  
  #Remove Company stopWords
  #corpus0 = tm_map(corpus0, removeWords,  companyStopWords)
  
  funs<-list(stripWhitespace , companyStopWords , stemDocument , stopWords , removePunctuation , tolower )
  corpus = tm_map(corpus , FUN=tm_reduce , tmFuns = funs)
  corpus
} 