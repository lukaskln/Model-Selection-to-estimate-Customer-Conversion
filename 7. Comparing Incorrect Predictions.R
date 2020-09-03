source(paste0(getwd(),"/ML_R_Code_Group_KLC/3. Feature Engineering.R")) # Transformed Dataset

# Comparing indices of incorrect prediction

dfValidation <- dfTrain_Data_Modeling[-c(1:6000),]
dfSubset <- dfTrain_Data_Modeling[1:6000,]

# fit the Optimal random forest model on training subset 

set.seed(123)
model.randomforest.compare <- randomForest(x = dfSubset[,-ncol(dfSubset)],
                                           y = dfSubset[,ncol(dfSubset)],
                                           ntree = 700, 
                                           mtry = 6, 
                                           nodesize=1,
                                           Importance = TRUE)

# make prediction on vaidation subset

yhat_train_rf <- predict(model.randomforest.compare, dfValidation[,-13], type = "class")

# Store true validation response and prediction 
dfYhats <- cbind(dfValidation$y, yhat_train_rf)

# extract indices of incorrect predictions with optimal random forest model

index.rf <- which(dfYhats[,1] != dfYhats[,2])


# fit the random forest model less education on training subset 
set.seed(123)
model.randomforest.noEdu.compare <- randomForest(x = dfSubset[,-c(3,ncol(dfSubset))],
                                                 y = dfSubset[,ncol(dfSubset)],
                                                 ntree = 700, 
                                                 mtry = 6, 
                                                 nodesize=1,
                                                 Importance = TRUE)

# make prediction 
yhat_noEdu <- predict(model.randomforest.noEdu.compare, dfValidation[,-c(3,13)], type = "class")

# store prediction 
dfYhats <- cbind(dfYhats,yhat_noEdu)

# Extract indices 
index.noEdu <- which(dfYhats[,1]!= dfYhats[,3])


# fit the optimal boosting model on training subset 
set.seed(123)
model.boosting.compare <- xgboost(model.matrix(y~.-1,dfSubset), as.integer(dfSubset$y)-1,
                                  nrounds = 1500,
                                  eta = 0.01,
                                  max_depth = 12,
                                  gamma = 7,
                                  colsample_bytree = 1,
                                  min_child_weight = 2,
                                  subsample = 0.75,
                                  objective = "binary:logistic",
                                  verbose = 0)

# make prediction 

matrix.validate<- model.matrix(y~.-1, dfValidation)
boost.predict <- as.numeric(c(predict(model.boosting.compare, matrix.validate) >= 0.5))+1

# store prediction 
dfYhats <- cbind(dfYhats, boost.predict)

# extract indices 
index.boost <- which(dfYhats[,1]!= dfYhats[,4])

# comparing indices (not symmetircal)

length(setdiff(index.boost, index.rf))
length(setdiff(index.rf, index.boost))
length(setdiff(index.rf, index.noEdu))
length(setdiff(index.noEdu, index.rf))
length(setdiff(index.boost, index.noEdu))
length(setdiff(index.noEdu,index.boost))



