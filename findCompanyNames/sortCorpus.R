sortCorpus <- function(corpus ){

	for ( i in 1:length(corpus)){
	corpus[i] = sort(corpus[i])
	}

	corpus

}