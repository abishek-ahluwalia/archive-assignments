mapDuplicates <-function(map , dup) {
  for (i in 1:length(dup)){
    map[i] = paste(map[i] , dup[i] , sep ="|")
  }
  
}