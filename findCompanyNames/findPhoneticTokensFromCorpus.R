
 findPhoneticTokensFromCorpus  <- function(corpus , num=5 , method="soundex"){
 phoneticCorpus = c()
 for ( i  in 1 : num) {
	newRecord = phonetic(as.character(corpus[[i]]))
	#print(newRecord)
	phoneticCorpus = append(phoneticCorpus, newRecord)
 }	
phoneticCorpus
}
