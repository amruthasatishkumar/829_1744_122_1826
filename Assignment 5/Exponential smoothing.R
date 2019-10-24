#time series
kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kingtimeseries <- ts(kings)
births <- scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")
#store birth as timeseries object collected per month (frequency =12)
birthstimeseries <- ts(births, frequency=12, start=c(1946,1))
#print(birthstimeseries)
souveneir <-scan("https://robjhyndman.com/tsdldata/data/fancy.dat")
soutimeseries <-ts(souveneir,frequency=12,start=c(1987,1))
plot.ts(birthstimeseries)
#plotting time series
plot.ts(kingtimeseries)
#plot.ts(birthtimeseries)
#decomposing non-seasonal data into trend and irregular component
#TTR library package for smoothing, SMA() function to calculate simple moving average decompose() function to decompose the data into seasonal,irregular and trend component
#Single Exponential Smoothing for data contating total annual rainfall(inch) in london from 1813-1912
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
print(rain)
rainseries <- ts(rain,frequency=1,start=c(1813,1))
print(rainseries)
plot.ts(rainseries)
rainseriesforecasts <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE)
print(rainseriesforecasts)
#print the forecasted values 
print(rainseriesforecasts$fitted)
plot(rainseriesforecasts)
HoltWinters(rainseries, beta=FALSE, gamma=FALSE, l.start=23.56)
library("forecast")
#prediction for annual rainfall for 1914-1920 using single exponential smoothing 
#installed forecast package using install.packages("forecast")
rainseriesforecasts2 <- forecast:::forecast.HoltWinters(rainseriesforecasts,h=8)
print(rainseriesforecasts2) #predicted interval values for years 1913-1920 with 80% and 95% prediction.
forecast:::plot.forecast(rainseriesforecasts2) #dark grey 80% prediction and light greay 80% prediction.

