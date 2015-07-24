 findPhoneticTokens  <- function(corpus , method="soundex"){
 phoneticCorpus = c()
 for ( i  in 1 :length(corpus)) {
  
	newRecord = phonetic( iconv(as.character(corpus[[i]]), "latin1", "ASCII", sub=""))

	phoneticCorpus = append(phoneticCorpus, newRecord)
 }	
phoneticCorpus
}