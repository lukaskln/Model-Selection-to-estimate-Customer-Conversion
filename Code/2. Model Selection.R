# Model overview on default dataset with 4 fold CV

source(paste0(getwd(),"/ML_R_Code_Group_KLC/1. Descriptive and Correlation Analysis.R")) # Load Data and libraries from EDA part, change path if needed. 
# Change / to \ for Mac and Linux


# LDA

k = 4 # Change K to the respective folds
set.seed(123) # To compare results, has to run always with sample() together!
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- lda(y ~., data = fold_train_data)
  
  pred <- as.numeric(predict(fit,fold_test_data)$class)-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# QDA

k = 4 
set.seed(123) 
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- qda(y ~., data = fold_train_data)
  
  pred <- as.numeric(predict(fit,fold_test_data)$class)-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# Logistic Regression

k = 4 
set.seed(123) 
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- glm(y ~ ., data = fold_train_data, family = binomial(link = "logit"))
  
  pred <- as.numeric(predict(fit,fold_test_data, type = "response") > 0.5)
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# Single Tree

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- rpart(y ~., data = fold_train_data, method = "class")
  
  pred <- as.numeric(predict(fit,fold_test_data, type = "class"))-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# Bagging

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- bagging(y~., data= fold_train_data, nbagg = 500)
  
  pred <- as.numeric(predict(fit,fold_test_data))-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# Random Forest

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- randomForest(y~., data= fold_train_data)
  
  pred <- as.numeric(predict(fit,fold_test_data))-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# Boosting

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- xgboost(model.matrix(y~.-1,fold_train_data), as.integer(fold_train_data$y)-1, # xgboost only takes data in sparse form
                 nrounds = 150, # Based on rule of thumb
                 objective = "binary:logistic",
                 verbose = 0) # no live training output in console
  
  pred <- as.numeric(predict(fit,model.matrix(y~.-1,fold_test_data)) > 0.5)
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# SVM

k = 4 
set.seed(123)
indices = sample(1:nrow(dfTrain_Data)) 
folds = cut(indices, breaks = k, labels = FALSE) 

vACC = c()

for (i in 1:k){ 
  
  cat("Processing fold #", i, "\n") 
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  
  fold_test_data <- dfTrain_Data[val_indices,] 
  
  fold_train_data <- dfTrain_Data[-val_indices,] 
  
  fit <- svm(formula = y ~ ., 
             data = fold_train_data, 
             type = 'C-classification', 
             kernel = 'radial')
  
  pred <- as.numeric(predict(fit,fold_test_data))-1
  
  vACC[i] <- 1 - mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
}

sum(vACC)/k 
sd(vACC)

# ANN 

model.ann <- keras_model_sequential() # Tensorflow backend needed

model.ann %>% 
  layer_dense(units = 32, activation = 'sigmoid', input_shape = dim(model.matrix(y~.-1,dfTrain_Data))[2]) %>%
  layer_dense(units = 16, activation = 'sigmoid') %>%
  layer_dense(units = 16, activation = 'sigmoid') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>% # Activation based on intuition
  compile(
    optimizer = 'rmsprop', # Optimizer choosen based on recommendation for binary classifications
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )

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
  
  model.ann %>% fit(model.matrix(y~.-1,fold_train_data), as.integer(fold_train_data$y)-1, epochs = 100, batch_size = 100)
  
  pred <- as.numeric(predict(model.ann,model.matrix(y~.-1,fold_test_data), type = "response" ) > 0.5)
  
  vACC[i] = 1- mean((c(as.integer(fold_test_data$y)-1) - pred)^2)
  
}

sum(vACC)/k
sd(vACC)


model.ann %>% fit(model.matrix(y~.-1,dfTrain_Data), as.integer(dfTrain_Data$y)-1, epochs = 200, batch_size = 100)

dfSparse_Test = model.matrix(ID ~.-1, dfTest_Data)

colnames(dfSparse_Test) = colnames(model.matrix(y ~.-1, dfTrain_Data))

pred <- as.numeric(c(predict(model.ann, dfSparse_Test) >= 0.5))

write.csv(data.frame(ID = 1:length(pred), y = pred), "Predictions.csv",row.names = FALSE)

