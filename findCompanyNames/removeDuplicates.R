library(tm)
library(SnowballC)
library(stringdist)
removeDuplicates <- function(names0 , corpus , uniqueNames, M){
  #remaining data
  D = corpus
  #uniqueNames
  C = uniqueNames
  b = deduplicationAlgoParamters
  
  i = 1

  if(4 %in% b){
    phoneticTokens = findPhoneticTokens(corpus)
    
    nameTokens = findPhoneticTokens(uniqueNames)
      
  }

  
  while(i <= length(C)){
    #variable to check if the word has been added to  
  
    #remove empty names
    while(is.na(C[i])||(C[i]=="")){
      #C = C[-i]
     # names0 = names0[-i]
    }
    j = 1
    while(j <= length(D)){
      #remove empty names  
      while(is.na(D[j])||(D[j]=="")){
        D = D[-j]
        names0 = names0[-j]
      }
      
      #variable to check if the jth name is still unmatched to ith name
      unMatched = 1
      
      
      #keyMatch
      if(1 %in% b){
        if(C[i]==D[j]){
          unMatched = 0
          M[i] = paste(M[i] , names0[j] , sep = "|")
          D = D[-j]
          names0 = names0[-j]
          j = j-1
        }   
      }
      
      #phonetic Match
      if((4 %in% b)&&(unMatched)){
        
        if(!stringdist(D[i] , D[j] , method="soundex")){
          unMatched = 0
          M[num] = paste(M[num] , names0[j] , sep = "|")
          D = D[-j]
          names0 = names0[-j]
          j = j-1
        }   
        
      }
      
      #levenshtein distance Match
      if((6 %in% b)&&(unMatched)){
        
        if(stringdist(D[i] , D[j] , method="lv")< editDistanceThreshold){
          if(!added){
            N = append(N , names0[i])
            M = append(M , names0[i])
            added = 1
            num = num +1
          }
          M[num] = paste(M[num] , names0[j] , sep = "|")
          D = D[-j]
          names0 = names0[-j]
          j = j-1
        }   
        
      }
      
      j = j+1
    }
    i = i+1
    
  }
  
  D1 = D
  DUP = M
  output = list("remainingData" = D1 ,"remainingNames" = names0, "duplicates" = DUP)
} 