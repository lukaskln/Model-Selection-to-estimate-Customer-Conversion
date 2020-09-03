# Boosting 

source(paste0(getwd(),"/ML_R_Code_Group_KLC/1. Descriptive and Correlation Analysis.R")) # Load Data and libraries from EDA part, change path if needed
source(paste0(getwd(),"/ML_R_Code_Group_KLC/3. Feature Engineering.R")) # Transformed Dataset

# Feature Selection

# By variable importance

model.boosting <- xgboost(model.matrix(y~.-1,dfTrain_Data), as.integer(dfTrain_Data$y)-1, 
                    nrounds = 150, 
                    objective = "binary:logistic",
                    verbose = 0) 

xgb.ggplot.importance(xgb.importance(colnames(model.matrix(y~.-1,dfTrain_Data)),model = model.boosting)) + 
  theme_minimal() + 
  coord_cartesian() + 
  scale_fill_manual(values = vColors[c(5,4,3,2,1)]) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# By stepwise in
  

dfEval = c()

for(j in 1:(ncol(dfTrain_Data)-1)){
  
  dfTrain_Data_Step <- dfTrain_Data[,c(1:j,ncol(dfTrain_Data))]

  k = 4 
  set.seed(123)
  indices = sample(1:nrow(dfTrain_Data_Step)) 
  folds = cut(indices, breaks = k, labels = FALSE) 
  
  vACC = c()
  
    for (i in 1:k){ 
      
      cat("Processing fold #",j, ".", i, "\n") 
      
      val_indices <- which(folds == i, arr.ind = TRUE)
      
      fold_test_data <- dfTrain_Data_Step[val_indices,] 
      
      fold_train_data <- dfTrain_Data_Step[-val_indices,] 
      
      fit <- xgboost(model.matrix(y~.-1,fold_train_data), as.integer(fold_train_data$y)-1, 
                     nrounds = 150, 
                     objective = "binary:logistic",
                     verbose = 0) 
      
      pred <- as.numeric(predict(fit,model.matrix(y~.-1,fold_test_data)) > 0.5)
      
      vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
    }
  
  nAvg = round(sum(vACC)/k,4)
  
  dfEval = rbind(dfEval,c(colnames(dfTrain_Data)[j],nAvg))
  
}

dfEval


# Hyperparameter optimization 

dfTrain_Data_Modeling = model.matrix(y~.-1, dfTrain_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4")))


# Search Grid for parameter combinations
xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(4,6,8,10),
  gamma = c(1,3,5,7,10),
  colsample_bytree =  c(0.4,0.6,0.8,1),
  min_child_weight = c(1,2,5,7),
  subsample = c(0.5,0.75,1)
)

# Conrol parameters
xgb_control = trainControl(
  method = "cv",
  number = 5, # Takes very long, change to 1 to lower the computation time (Still around one hour)
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        
  classProbs = TRUE,                                                           
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)


levels(dfTrain_Data$y) = c("no","yes") # Caret can not work with numbers as binary classifier

# Train the model for each parameter combination in the grid

xgb_train = train(
  x = dfTrain_Data_Modeling,
  y = dfTrain_Data$y,
  trControl = xgb_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

levels(dfTrain_Data$y) = c(0,1) # Set levels back

# Finetuning with 5 fold CV

dfTrain_Data_Modeling = dfTrain_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4"))

nMin = c()

cv.res <- xgb.cv(model.matrix(y~.-1,dfTrain_Data_Modeling),label = as.integer(dfTrain_Data_Modeling$y)-1, # best model by hyperparameter finetuning
                 params = list(
                   nrounds = 1500,
                   eta = 0.01,
                   max_depth = 12,
                   gamma = 6.8,
                   colsample_bytree = 1,
                   min_child_weight = 2,
                   subsample = 0.75,
                   objective = "binary:logistic",
                   verbose = 2),
                 seed = 123,
                 nfold = 5,
                 nrounds = 2000)

it = which.min(cv.res$evaluation_log$test_error_mean)
nMin.old = nMin
best.iter = cv.res$evaluation_log$iter[it]
nMin = min(cv.res$evaluation_log$test_error_mean)
nMin.old
nMin

plot(cv.res$evaluation_log$test_error_mean, type = "l", ylim =c(0.1,0.18), col = "Red")
lines(cv.res$evaluation_log$train_error_mean, col = "Blue")


# Evaluation of optimal model

# 4 Fold CV

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data_Modeling)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data_Modeling[val_indices,] 
  
  fold_train_data <- dfTrain_Data_Modeling[-val_indices,] 
  
  fit <- xgboost(model.matrix(y~.-1,fold_train_data), as.integer(fold_train_data$y)-1,
                 nrounds = 1500,
                 eta = 0.01,
                 max_depth = 12,
                 gamma = 6.8,
                 colsample_bytree = 1,
                 min_child_weight = 2,
                 subsample = 0.75,
                 objective = "binary:logistic",
                 verbose = 0)
  
  pred <- as.numeric(predict(fit,model.matrix(y~.-1,fold_test_data)) > 0.5)
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
  
  print(table(as.integer(fold_test_data$y)-1,pred))
}

sum(vACC)/k
sd(vACC)


# Importance Plot

model.boosting <- xgboost(model.matrix(y~.-1,dfTrain_Data_Modeling), as.integer(dfTrain_Data_Modeling$y)-1,
                          nrounds = 1200,
                          eta = 0.01,
                          max_depth = 12,
                          gamma = 6.8,
                          colsample_bytree = 1,
                          min_child_weight = 2,
                          subsample = 0.75,
                          objective = "binary:logistic",
                          verbose = 0)


xgb.ggplot.importance(xgb.importance(colnames(model.matrix(y~.-1,dfTrain_Data_Modeling)),model = model.boosting)) + 
  theme_minimal() + 
  coord_cartesian() + 
  scale_fill_manual(values = vColors[c(5,4,3,2,1)]) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Test data prediction

dfTest_Data_Modeling <- dfTest_Data %>% dplyr::select(-c("X2","job","banner_views_old","X4")) 

dfSparse_Test = model.matrix(ID ~.-1, dfTest_Data_Modeling)

colnames(dfSparse_Test) = colnames(model.matrix(y ~.-1, dfTrain_Data_Modeling))

pred <- as.numeric(c(predict(model.boosting, dfSparse_Test) >= 0.5))

write.csv(data.frame(ID = 1:length(pred), y = pred), "Predictions.csv",row.names = FALSE)



















