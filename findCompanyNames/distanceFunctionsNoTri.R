findDistance <- function( a, b , method ){

 distance = stringdist(a,b, method = method)
 distance
}

source

 findDistancesFromCorpus <- function(corpus , method){
distances = c()
 for ( i  in 1 : length(corpus)) {
	#extract string from plaintextdocument from corpus
	a = as.character(corpus[[i]])
	dist = c()
	for( j in 1:length(corpus) ){
		b = as.character(corpus[[j]])
		dist = append(dist , findDistance( a, b , method ))
	}
	distances = rbind.data.frame(distances , dist)
 }	
distances
}

#run for soundex and lv 
#distances = findDistancesFromCorpus(corpus0 , method =  "soundex")?