# Random Forest

source(paste0(getwd(),"/ML_R_Code_Group_KLC/1. Descriptive and Correlation Analysis.R")) # Load Data and libraries from EDA part, change path if needed
source(paste0(getwd(),"/ML_R_Code_Group_KLC/3. Feature Engineering.R")) # Transformed Dataset

# Feature selection by Importance measure
boruta_output <- Boruta(y ~ ., data= dfTrain_Data, doTrace=2)  
names(boruta_output)


boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE) # Get significant variables including tentatives
print(boruta_signif) 

plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

# Stepwise in Variable selection

dfEval = c()

for(j in 1:(ncol(dfTrain_Data)-1)){
  
  dfTrain_Data_Step <- dfTrain_Data[,c(1:j,ncol(dfTrain_Data))]
  
  k = 4 
  set.seed(123) 
  indices = sample(1:nrow(dfTrain_Data)) 
  folds = cut(indices, breaks = k, labels = FALSE) 
  
  vACC = c()
  
  for (i in 1:k){ 
    
    cat("Processing fold #",j, ".", i, "\n") 
    
    val_indices = which(folds == i, arr.ind = TRUE)
    
    fold_test_data = dfTrain_Data_Step[val_indices,] 
    
    fold_train_data = dfTrain_Data_Step[-val_indices,] 
    
    fit <- randomForest(y ~ ., data = fold_train_data, ntree = 500, mtry = min(c(j,9)), importance = TRUE)
    
    pred <- as.numeric(predict(fit, fold_test_data, type = "class"))-1
    
    vACC[i] = 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
  }
  
  nAvg = round(sum(vACC)/k,4)
  
  dfEval = rbind(dfEval,c(colnames(dfTrain_Data)[j],nAvg))
  
}

dfEval

# mtry hyperparameter tuning 
mtry_4 <- randomForest(y ~ ., data = dfTrain_Data, importance = TRUE)
mtry_6 <- randomForest(y ~ ., data = dfTrain_Data, ntree = 500, mtry = 6, importance = TRUE)

par(mfrow = c(1,2))
plot(mtry_4, lwd = 2)
plot(mtry_6, lwd = 2)

par(mfrow = c(1,1))
varImpPlot(mtry_6)

# With tuneRF function

set.seed(123)

mtry.tuneRF <- tuneRF(subset(dfTrain_Data, select = -y), dfTrain_Data$y, stepFactor=1.5, improve=1e-5, ntree=700)

mtry.tuneRF

# We used different combinations of ntree, mtry and nodesize on the original 
# dataset to check for CV accuracy and the sd of the ACC to see how consistent 
# the prediction performed across all folds.  Best model with: ntree = 700, mtry = 6, nodesize=1.

k = 4 
set.seed(123) 
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices = which(folds == i, arr.ind = TRUE)
  
  fold_test_data = dfTrain_Data[val_indices,] 
  
  fold_train_data = dfTrain_Data[-val_indices,] 
  
  fit <- randomForest(y ~ ., data = fold_train_data, ntree = 700, mtry = 6, nodesize=1, importance = TRUE)
  
  pred <- as.numeric(predict(fit, fold_test_data, type = "class"))-1
  
  vACC[i] = 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)


# Performance on the transformed Dataset, larger node size (10) gives better results

dfTrain_Data_Modeling = dfTrain_Data_Transformed 

k = 4 
set.seed(123) 
indices = sample(1:nrow(dfTrain_Data_Modeling)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices = which(folds == i, arr.ind = TRUE)
  
  fold_test_data = dfTrain_Data_Modeling[val_indices,] 
  
  fold_train_data = dfTrain_Data_Modeling[-val_indices,] 
  
  fit <- randomForest(y ~ ., data = fold_train_data, ntree = 700, mtry = 6, nodesize = 10)
  
  pred <- as.numeric(predict(fit, fold_test_data, type = "class"))-1
  
  vACC[i] = 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)



# Dropping X2, job, banner_views_old and X4 as recommended in the feature selection but no transformation

dfTrain_Data_Modeling = dfTrain_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4"))

k = 4 
set.seed(123) 
indices = sample(1:nrow(dfTrain_Data_Modeling)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices = which(folds == i, arr.ind = TRUE)
  
  fold_test_data = dfTrain_Data_Modeling[val_indices,] 
  
  fold_train_data = dfTrain_Data_Modeling[-val_indices,] 
  
  fit <- randomForest(y ~ ., data = fold_train_data, ntree = 700, mtry = 6, nodesize = 1)
  
  pred <- as.numeric(predict(fit, fold_test_data, type = "class"))-1
  
  vACC[i] = 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
  
  print(table(as.integer(fold_test_data$y)-1, pred)) # Confusion matrix per fold
}

sum(vACC)/k 
sd(vACC)

# Test dataset performance

dfTrain_Data_Modeling = dfTrain_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4"))
dfTest_Data_Modeling = dfTest_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4"))

model.randomforest <- randomForest(x = dfTrain_Data_Modeling[,-ncol(dfTrain_Data_Modeling)],
                    y = dfTrain_Data_Modeling[,ncol(dfTrain_Data_Modeling)],
                    ntree = 700, 
                    mtry = 6, 
                    nodesize=1,
                    Importance = TRUE)

yhat <- predict(model.randomforest, dfTest_Data_Modeling[,-1], type="class")


write.csv(data.frame(ID=1:nrow(dfTrain_Data_Modeling), y =yhat), file='PredictionRF.csv', row.names=FALSE)

# Importance Plot of final model

varImpPlot(model.randomforest)
