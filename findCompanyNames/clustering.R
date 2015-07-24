
library(tm)
library(SnowballC)
library(stringdist)
library(stringr)
clustering <- function(corpus , names0 ){
  D = corpus
  #uniqueKeys
  K = c()
  #uniqueNames
  N = c()
  #map ofduplicates
  M = c()
  #number of uniqueNames seen yet
  num = 0
  
  b = clusteringAlgoParameters
  
  i = 1
  
  
  if((5 %in% b)&&(unMatched)){
    
  }
  
  
  while(i <= length(D)){
    #variable to check if the word has been added to  
    added = 0 
    j = i +1
    
    #remove empty names
    while((is.na(D[i])||D[i]=="")){
      D = D[-i]
      names0 = names0[-i]
    }
        
    while(j <= length(D)){
      #remove empty names  
      while(is.na(D[i])||(D[j]=="")){
        D = D[-j]
        names0 = names0[-j]
      }
      
      #variable to check if the jth name is still unmatched to ith name
      unMatched = 1
      
      
      #keyMatch
      if(1 %in% b){
       if(D[i]==D[j]){
        unMatched = 0
        if(!added){
          N = append(N , names0[i])
          K = append(K , D[i])
          M = append(M , names0[i])
          added = 1
          num = num +1
        }
        print(-num)
        M[num] = paste(M[num] , names0[j] , sep = "|")
        D = D[-j]
        names0 = names0[-j]
        j = j-1
       }   
      }
 
      #phonetic Match Soundex
      if((4 %in% b)&&(unMatched)){
        
        if(!stringdist(D[i] , D[j] , method="soundex")){
          if(!added){
            N = append(N , names0[i])
            M = append(M , names0[i])
            added = 1
            num = num +1
          }
          print(num)
          M[num] = paste(M[num] , names0[j] , sep = "|")
          D = D[-j]
          names0 = names0[-j]
          j = j-1
        }   
        
      }
  
      
      if((5 %in% b)&&(unMatched)){
        
        if(!stringdist(D[i] , D[j] , method="soundex")){
          if(!added){
            N = append(N , names0[i])
            M = append(M , names0[i])
            added = 1
            num = num +1
          }
          print(num)
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
          print(num)
          M[num] = paste(M[num] , names0[j] , sep = "|")
          D = D[-j]
          names0 = names0[-j]
          j = j-1
        }   
    
      }
      
      j = j+1
     }
    if(!added){
      N = append(N , names0[i])
      K = append(K , D[i])
      M = append(M , names0[i])
      num = num +1
      
    }
    i = i+1
     
   }
  print (num)
output = list("uniqueNames" = N , "duplicates" = M , "uniqueKeys" = K)

output
} 