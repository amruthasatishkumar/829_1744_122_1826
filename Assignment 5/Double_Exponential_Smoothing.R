#dataset: diameter of women skirts produced at hem annually from 1866-1911
skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5)
print(skirts)
#create a time series object
skirtsseries <- ts(skirts,start=c(1866)) #create a time series model
plot.ts(skirtsseries)
#Perform Double Exponential Smoothing
skirtsseriesforecasts <- HoltWinters(skirtsseries, gamma=FALSE) 
print(skirtsseriesforecasts) #gives the value of alpha and beta parameters
plot(skirtsseriesforecasts) #plot forecasted results
#perform prediction for years 1912-1930
skirtsseriesforecasts2 <- forecast ::: forecast.HoltWinters(skirtsseriesforecasts, h=19)
print(skirtsseriesforecasts2)
forecast:::plot.forecast(skirtsseriesforecasts2) #light grey 95% prediction , #dark grey 90% prediction
