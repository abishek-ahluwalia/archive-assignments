

findDistance <- function( a, b , method ){

 distance = stringdist(a,b, method = method)
 distance
}


 findDistancesFromCorpusTri <- function(corpus , method){
#store symmetric matrix as triangular matrix in a vector
dist = c()
 for ( i  in 1 : length(corpus)) {
	#extract string from plaintextdocument from corpus
	a = as.character(corpus[[i]])
	for( j in 1: i ){
		b = as.character(corpus[[j]])
		dist = append(dist , findDistance( a, b , method ))
	}
 }	
dist
}

#triangular matrix vector
#distances = findDistancesFromCorpusTri(corpus0 , method = "soundex")

#get distacne between ith and jth element in corpus from triangular matrix vector
getDistance <- function(i , j , method){
	#ith row begins after --> i(i-1)/2
	#jth element in ith row ---> i(i-1)/2 + j
	distances[i*(i-1)/2 + j]

} 


