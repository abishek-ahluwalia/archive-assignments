

#read Data
#names0 = readLines("names.txt")
#dataPreprocessing 
#if a user is manually going to check the stop words as recommended then it 
#should be done in between this dataPreprocessing step manually instead of 
#running this command directly

#
#removePunctuation <- function(x) UseMethod("removePunctuation", x)
#removePunctuation.PlainTextDocument <- function(x) gsub("[[:punct:]]+",
 #                                                       " ", x)

#corpus0 = dataPreprocessing(names0)
#data = createCorpus(corpus0)

corpus = data$corpus
stopWords = data$stop



#find names 

#if a user is going to manually validate all the unique company names found 
#in each iteration then it should be done after each each iteration of the 
#following loop is finished instead of running it automatically like this 

start = 1
end = chunkSize
allUniqueNames = c()
duplicates = c()
while (length(corpus) >0){
  #split a chunk
  if( end >=length(corpus)) end = length(corpus)
  X = corpus[start:end]
  Y = names0[start:end]

  #clustera
  companyNames = clustering(X , Y )
  N = companyNames$uniqueNames 
  M = companyNames$duplicates
  K = companyNames$uniqueKeys
  
  #append to unique names
  allUniqueNames = append(allUniqueNames , N)
  
  if( end <length(corpus)){
  corpus = corpus[-start:-end]
  names0 =names0[-start:-end]
  #findDuplicates in rest of the data
  output = removeDuplicates(names0, corpus , K  , M)

  corpus=output$remainingData
  names0 = output$remainingNames
  M = output$duplicates
#  M = mapDuplicates(M , DUP)
  }else{
    corpus = c()
    names0 = c()
  }
  duplicates = append(duplicates , M)
  
} 
  
