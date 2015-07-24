convertCorpus <- function(corpus){
  newCorpus = c()
  for ( i  in 1 : length(corpus)) {
    #extract string from plaintextdocument from corpus
    a = as.character(corpus[[i]])
    newCorpus = append(newCorpus , a)
    
  }
  newCorpus
}