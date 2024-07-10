##taking train and test data setts
EGX30_TRAIN<-EGX30_2010_now[1431:2892,]
EGX30_Test<-EGX30_2010_now[2893:nrow(EGX30_2010_now),]

EGX70_TRAIN<-EGX70_Indices[1:974,]
EGX70_Test<-EGX70_Indices[975:nrow(EGX70_Indices),]

TAMAYUZ_Train<-TAMAYUZ_Nilex_Indices[1:974,]
TAMAYUZ_Test<-TAMAYUZ_Nilex_Indices[975:nrow(TAMAYUZ_Nilex_Indices),]

##############################################################################################################################
####for EGX30
##descriptive measures 
library(ggplot2)
library(tseries)
library(forecast)
library(lmtest)
summary(EGX30_2010_now$INDEXCLOSE)

##boxplot 
box1<-ggplot(data=EGX30_2010_now,aes(y=INDEXCLOSE,x=""))+
  geom_boxplot(fill = "green",alpha=0.5,outlier.colour ="red",colour="#4B0082")+ 
  ggtitle("EGX30_TRAIN Boxplot") + theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
# Create the time series plot and acf plot and pacf plot 
plot1<-ggplot(EGX30_2010_now, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "EGX-30", title = "Time Series Plot for EGX-30")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

acf(EGX30_TRAIN$INDEXCLOSE,main = "Autocorrelation Function Plot for EGX-30")
pacf(EGX30_TRAIN$INDEXCLOSE,main = "partial Autocorrelation Function Plot for EGX-30")

##ducky filler test of stationary 

adf.test(EGX30_TRAIN$INDEXCLOSE) ## non stationary 

##kruskal.test test of seasonality 

kruskal.test(EGX30_TRAIN$INDEXCLOSE~EGX30_TRAIN$INDEXDATE, data = EGX30_TRAIN)


######################################################################################################################
####Egx-70
######descriptive measures
summary(EGX70_Indices$INDEXCLOSE)

##boxplot 
box2<-ggplot(data=EGX70_Indices,aes(y=INDEXCLOSE,x=""))+
  geom_boxplot(fill = "green",alpha=0.5,outlier.colour ="red",colour="#4B0082")+ 
  ggtitle("EGX70 Boxplot") + theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
# Create the time series plot and acf plot and pacf plot 
plot2<-ggplot(EGX70_Indices, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "EGX-70", title = "Time Series Plot for EGX-70")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

acf(EGX70_Indices$INDEXCLOSE,main = "Autocorrelation Function Plot for EGX-70")
pacf(EGX70_Indices$INDEXCLOSE,main = "partial Autocorrelation Function Plot for EGX-70")

##ducky filler test of stationary 

adf.test(EGX70_TRAIN$INDEXCLOSE) ## non stationary 

##kruskal.test test of seasonality 

kruskal.test(EGX70_TRAIN$INDEXCLOSE~EGX70_TRAIN$INDEXDATE, data = EGX70_TRAIN)

##################################################################################################################################
####for tamyz
######descriptive measures
summary(TAMAYUZ_Train$INDEXCLOSE)

##boxplot 
box3<-ggplot(data=TAMAYUZ_Nilex_Indices,aes(y=INDEXCLOSE,x=""))+
  geom_boxplot(fill = "green",alpha=0.5,outlier.colour ="red",colour="#4B0082")+ 
  ggtitle("TAMAYUZ Boxplot") + theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))
# Create the time series plot and acf plot and pacf plot 
plot3<-ggplot(TAMAYUZ_Nilex_Indices, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "TAMAYUZ", title = "Time Series Plot for TAMAYUZ")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

acf(TAMAYUZ_Train$INDEXCLOSE,main = "Autocorrelation Function Plot for TAMAYUZ")
pacf(TAMAYUZ_Train$INDEXCLOSE,main = "partial Autocorrelation Function Plot for TAMAYUZ")

##ducky filler test of stationary 

adf.test(TAMAYUZ_Train$INDEXCLOSE) ## non stationary 

##kruskal.test test of seasonality 

kruskal.test(TAMAYUZ_Train$INDEXDATE~TAMAYUZ_Train$INDEXCLOSE, data = TAMAYUZ_Train)

########################################################################################################
library(cowplot)
combined_plots_1 <- plot_grid( box1,box2,box3, ncol = 3)
combined_plots_2 <- plot_grid( plot1,plot2,plot3, nrow = 3)
###############################################################################################
##time series plots for taining data sets 
ggplot(EGX30_TRAIN, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "EGX-30", title = "Time Series Plot for EGX-30")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggplot(EGX70_TRAIN, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "EGX-70", title = "Time Series Plot for EGX-70")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggplot(TAMAYUZ_Train, aes(x = INDEXDATE, y = INDEXCLOSE)) +
  geom_line() +
  labs(x = "Date", y = "TAMAYUZ index", title = "Time Series Plot for TAMAYUZ")+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                                                                                            panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


######################################################################################################################################################
###making series stationary 
###detrending using first difference && fitting arima  
##EGX30 
summary(EGX30_TRAIN$INDEXCLOSE)##before ln 

loG_EGX30<-as.ts(log(EGX30_TRAIN$INDEXCLOSE))##taking ln 
summary(log(EGX30_TRAIN$INDEXCLOSE))##after ln

first_diff_EGX30<-diff(loG_EGX30)##first diff
plot.ts(first_diff_EGX30)
acf(first_diff_EGX30)
pacf(first_diff_EGX30,lag.max = 20)

close_price_EGX30 <- ts(data = EGX30_TRAIN$INDEXCLOSE)
ARIMA_EGX30<-auto.arima(y = loG_EGX30, seasonal = FALSE, trace = TRUE)##arima model
coeftest(ARIMA_EGX30)## sig, of parameters 

acf(ARIMA_EGX30$residuals)
pacf(ARIMA_EGX30$residuals)



pvalue_egx30<-c()
lags<-seq(1,200,1)
for (i in 1:length(lags)) {m<-Box.test(ARIMA_EGX30$residuals, lag=lags[i], type="Ljung-Box")
pvalue_egx30[i]=m$p.value}



##EGX70
summary(EGX70_TRAIN$INDEXCLOSE)#before ln

loG_EGX70<-as.ts(log(EGX70_TRAIN$INDEXCLOSE))# taking ln trans.
summary(log(EGX70_TRAIN$INDEXCLOSE))##after ln 

first_diff_EGX70<-diff(loG_EGX70)#taking 1st diff
plot.ts(first_diff_EGX70)

adf.test(first_diff_EGX70) # ducky test 
acf(first_diff_EGX70)
pacf(first_diff_EGX70,lag.max = 20)

close_price_EGX70 <- ts(data = EGX70_TRAIN$INDEXCLOSE)
ARIMA_EGX70<-auto.arima(y = loG_EGX70, seasonal = FALSE, trace = TRUE)##arima model
coeftest(ARIMA_EGX70)## sig, of parameters 

acf(ARIMA_EGX70$residuals)
pacf(ARIMA_EGX70$residuals)




pvalue_egx70<-c()
lags<-seq(1,100,1)
for (i in 1:length(lags)) {m<-Box.test(ARIMA_EGX70$residuals, lag=lags[i], type="Ljung-Box")
pvalue_egx70[i]=m$p.value}



##Tamyz
summary(TAMAYUZ_Train$INDEXCLOSE)##before ln 

loG_Tamyz<-as.ts(log(TAMAYUZ_Train$INDEXCLOSE)) # taking ln trans.
summary(log(TAMAYUZ_Train$INDEXCLOSE))

first_diff_Tamyz<-diff(loG_Tamyz)#taking 1st diff
plot.ts(first_diff_Tamyz)##after ln

adf.test(first_diff_Tamyz) # ducky test 
acf(first_diff_Tamyz,lag.max = 20)
pacf(first_diff_EGX70,lag.max = 20)

close_price_Tamyz <- ts(data = TAMAYUZ_Train$INDEXCLOSE)
ARIMA_Tamyz<-auto.arima(y = loG_Tamyz, seasonal = FALSE, trace = TRUE)##arima model 
model_Tamyz <- arima(loG_Tamyz, order = c(0,1,1))
coeftest(model_Tamyz) ## sig, of parameters 

acf(model_Tamyz$residuals)
pacf(model_Tamyz$residuals)





pvalue_Tamyz<-c()
lags<-seq(1,100,1)
for (i in 1:length(lags)) {m<-Box.test(model_Tamyz$residuals, lag=lags[i], type="Ljung-Box")
pvalue_Tamyz[i]=m$p.value}




###########################################################################################################
####forecating 
##EGX-30
EGX30_Test$INDEXCLOSE<-log(EGX30_Test$INDEXCLOSE)
forecast_EGX30<-forecast(ARIMA_EGX30,h=500)
plot.ts(loG_EGX30)
plot(forecast_EGX30)
checkresiduals(forecast_EGX30)

##EGX-70
EGX70_Test$INDEXCLOSE<-log(EGX70_Test$INDEXCLOSE)
forecast_EGX70<-forecast(ARIMA_EGX70, h=282)
plot.ts(loG_EGX70)
plot(forecast_EGX70)
checkresiduals(forecast_EGX70)


##Tamyz
TAMAYUZ_Test$INDEXCLOSE<-log(TAMAYUZ_Test$INDEXCLOSE)
forecast_Tamyz<-forecast(model_Tamyz, h=281)
plot.ts(TAMAYUZ_Test$INDEXCLOSE)
plot(forecast_Tamyz)
checkresiduals(forecast_Tamyz)






