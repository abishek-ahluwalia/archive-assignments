#parameters 
#stopFreqThreshold as a thumb rule try threshold freq from 0.5 % to 2% 
#remove any company and city names from this list

findCompanyStopWords<- function(corpus0 , stopFreqThreshold = thresholdFrequencyStopWords){
#create ferquency matrix 
freq = DocumentTermMatrix(corpus0)


# Check for sparsity

stop10 = findFreqTerms(freq, lowfreq=10)
stop20 = findFreqTerms(freq, lowfreq=20)



companyStopWords = findFreqTerms(freq, lowfreq=stopFreqThreshold)
output = list("freq" = freq , "stop" = companyStopWords)
output
}