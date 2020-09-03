source(paste0(getwd(),"/ML_R_Code_Group_KLC/1. Descriptive and Correlation Analysis.R")) # Load Data and libraries from EDA part, change path if needed


# feature transformation 

dfTrain_Data_Transformed = dfTrain_Data

# log transformation of time_spent

dfTrain_Data_Transformed = dfTrain_Data_Transformed %>% mutate(log_time_spent = log(time_spent + 1))

par(mfrow = c(1,2))
hist(dfTrain_Data_Transformed$time_spent, main = "Histogram of time_spent", xlab = "time_spent")
hist(dfTrain_Data_Transformed$log_time_spent, main = "Histogram of log(time_spent + 1)", xlab = "log of time_spent")

# month as factor

dfTrain_Data_Transformed = dfTrain_Data_Transformed %>% mutate(month = factor(month))

# bucketing 

par(mfrow = c(1,2))
hist(dfTrain_Data_Transformed$days_elapsed_old, main = "Histogram of days_elapsed_old", col = vColors[2], border = n, axes = T, xlab = "days_elapsed_old")
abline(v = 0, col = vColors[1], lwd = 1.5)
abline(v = 200, col = vColors[1], lwd = 1.5)
abline(v = 400, col = vColors[1], lwd = 1.5)
hist(dfTrain_Data_Transformed$X4, breaks = 200, main = "Histogram of X4", col = vColors[2], border = n, axes = T, xlab = "X4", xlim = c(0,0.5))
abline(v = 0.07, col = vColors[1], lwd = 1.5)
abline(v = 0.075, col = vColors[1], lwd = 1.5)


dfTrain_Data_Transformed$days_elapsed_old_bucket = cut2(dfTrain_Data_Transformed$days_elapsed_old, c(0,200,400))
dfTrain_Data_Transformed$banner_views_old0 = ifelse(dfTrain_Data_Transformed$banner_views_old ==0, 1, 0)
dfTrain_Data_Transformed$banner_views_old0 = factor(as.character(dfTrain_Data_Transformed$banner_views_old0), levels = c("0", "1"))


table_days = table(dfTrain_Data$day)
par(mfrow = c(1,1))
barchart(table_days, col = vColors[1], horizontal = F, cex.names = 0.5, xlab = "Day of the Month", main = "Count of Observation by Variable 'Day'")

vDays = as.vector(table_days)
summary(vDays)

dfTrain_Data_Transformed$day_lowhigh = ifelse(dfTrain_Data_Transformed$day %in% c(1,20,24,31), 1,0)
dfTrain_Data_Transformed$day_lowhigh = factor(as.character(dfTrain_Data_Transformed$day_lowhigh), levels = c("0", "1"))



# 2. Categorical features 
dfTrain_Data$job %>% summary()

# 2.1 reordering eduction 

levels(dfTrain_Data$education)

dfTrain_Data_Transformed$education_ordered = ordered(dfTrain_Data_Transformed$education, levels = c("high_school", "university", "na","grad_school"))


# job 

dfTrain_Data_Transformed$job_housebound = ifelse(dfTrain_Data_Transformed$job %in% c("retired", "freelance", "housekeeper"), 1,0)

dfTrain_Data_Transformed$job_housebound = factor(as.character(dfTrain_Data_Transformed$job_housebound), levels = c("0", "1"))


# device 

dfTrain_Data_Transformed$device_binary = ifelse(dfTrain_Data_Transformed$device == "na", 0,1)
dfTrain_Data_Transformed$device_binary = factor(as.character(dfTrain_Data_Transformed$device_binary), levels = c("0", "1"))

# new dataframe 

dfTrain_Data_Transformed = dfTrain_Data_Transformed %>% dplyr::select(-c("job","banner_views_old","X2","X4"))

dfTrain_Data_Transformed %>% summary()


