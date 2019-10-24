souvenir <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987,1))
print(souvenirtimeseries)
logsouvenirtimeseries <- log(souvenirtimeseries) #we take log because of increased seasonal and random fluctutaions additive model cannot capture time series properly
plot(logsouvenirtimeseries)
#Perform Triple Exponential Smoothing, For the data consisting seasonal component
souvenirtimeseriesforecasts <- HoltWinters(logsouvenirtimeseries)
print(souvenirtimeseriesforecasts) #prints the alpha,beta and gamma parameters.. 
plot(souvenirtimeseriesforecasts) # plot forecasted values 
#Make prediction for for January 1994 to December 1998 , total 48 months 
souvenirtimeseriesforecasts2 <- forecast ::: forecast.HoltWinters(souvenirtimeseriesforecasts, h=48)
print(souvenirtimeseriesforecasts2) #print predicted values 80% predicted interval and 95% predicted interval

#Plot for predicted values 
forecast:::plot.forecast(souvenirtimeseriesforecasts2) #Blue line indicated predicted values 
