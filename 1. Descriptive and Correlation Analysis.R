# Packages 

library(tidyverse)
library(ggExtra)
library(GGally)
library(ggthemes)
library(gridExtra)
library(grid)
library(corrplot)
library(randomForest)
library(xgboost)
library(visNetwork)
library(rpart)
library(ipred)
library(tensorflow) # Python 3 has to be installed
library(keras) # Tensorflow on Python or CNTK on C++ has to be installed 
library(igraph)
library(Matrix)
library(e1071)
library(MASS)
library(Hmisc)
library(tuneRanger)
library(ggmosaic)
library(ISLR)
library(tree)
library(Boruta)
library(caret)

# Design Aspects

vColors <- c("#010B40","#F25C05", "#F29F05", "#626473", "#F27405")

theme_set(theme_minimal())

# Data Import

dfTrain_Data <- read.csv("train.csv") # indicator in front of objects indicates type/class of the object. df i.e. is data.frame 

dfTrain_Data <- dfTrain_Data %>% mutate(y = factor(y), X1 = factor(X1), X2 = factor(X2), X3 = factor(X3)) # Change integers to factors

## Explorative Data Analysis

## Descriptive Analysis

# Continuous

dfTrain_Data %>% select_if(is.numeric) %>% summary()

# Boxplots

vCol_names_1 = dfTrain_Data %>% select_if(is.numeric) %>% colnames()

lPlot_list_2 = list()

  
for(i in 1:length(vCol_names_1)){
  
  data = dfTrain_Data %>% dplyr::select(y,vCol_names_1[i]) # select() is used by several packages 
  
  colnames(data) = c("X1", "X2")
  
  lPlot_list_2[[i]] <- local({
  i <- i
  p1 <- ggplot(data, aes(x = X1, y = X2)) + geom_boxplot(color = vColors[1:2], fill = vColors[1:2], alpha = 0.4) + 
    xlab("Response") + ylab(paste(vCol_names_1[i])) +
    labs(title = paste0(i,". ",vCol_names_1[i]))
  })
  
}

do.call("grid.arrange", c(lPlot_list_2, ncol=4)) 
  

# Discrete 

dfTrain_Data %>% select_if(is.factor) %>% summary()

# Bar Charts

vCol_names_2 = dfTrain_Data %>% select_if(is.factor) %>% colnames()

lPlot_list_2 = list()


for(i in 1:(length(vCol_names_2)-1)){
  
  data = dfTrain_Data %>% dplyr::select(y,vCol_names_2[i])  
  
  colnames(data) = c("X1", "X2")
  
  lPlot_list_2[[i]] <- local({
    i <- i
    p1 <- ggplot(data, aes(y = X2, fill = X1)) + geom_bar(alpha = 0.8) + 
      ylab(paste(vCol_names_2[i])) +
      scale_fill_manual(values = vColors[c(1,2)]) +
      scale_color_manual(values = vColors[c(1,2)]) +
      guides(fill=guide_legend(title="Response"))  +
      labs(title = paste0(i,". ",vCol_names_2[i])) + theme(legend.position = "none")
  })
  
  
}

lPlot_list_2[[8]] = lPlot_list_2[[8]] + theme(legend.position = "right")
do.call("grid.arrange", c(lPlot_list_2, ncol=4)) 


# Mosaic Plots

dfTrain_Data$device = factor(dfTrain_Data$device,levels(dfTrain_Data$device)[c(2,1,3)]) # Changing Factor order only for this analysis
dfTrain_Data$y = factor(dfTrain_Data$y,levels(dfTrain_Data$y)[c(2,1)])

p1 =
  ggplot(data = dfTrain_Data) +
  geom_mosaic(aes(x = product(device), fill= y), na.rm = TRUE) + 
  scale_fill_manual(values = vColors[c(2,1)]) +
  scale_color_manual(values = vColors[c(2,1)]) +
  guides(fill=guide_legend(title="Response")) +
  labs(title = "1. Device") + theme(legend.position = "none")

dfTrain_Data$job = factor(dfTrain_Data$job,levels(dfTrain_Data$job)[rev(c(9,7,12,5,2,10,11,6,1,8,3))])

p2 =
  ggplot(data = dfTrain_Data) +
  geom_mosaic(aes(x = product(job), fill= y), na.rm = TRUE) + 
  scale_fill_manual(values = vColors[c(2,1)]) +
  scale_color_manual(values = vColors[c(2,1)]) +
  guides(fill=guide_legend(title="Response")) +
  labs(title = "5. Job") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

dfTrain_Data$education = factor(dfTrain_Data$education,levels(dfTrain_Data$education)[c(2,4,3,1)])

p3 =
  ggplot(data = dfTrain_Data) +
  geom_mosaic(aes(x = product(education), fill= y), na.rm = TRUE) + 
  scale_fill_manual(values = vColors[c(2,1)]) +
  scale_color_manual(values = vColors[c(2,1)]) +
  guides(fill=guide_legend(title="Response")) +
  labs(title = "2. Education") + theme(legend.position = "none")

dfTrain_Data$outcome_old = factor(dfTrain_Data$outcome_old,levels(dfTrain_Data$outcome_old)[c(2,1,3,4)])

p4 =
  ggplot(data = dfTrain_Data) +
  geom_mosaic(aes(x = product(outcome_old), fill= y), na.rm = TRUE) + 
  scale_fill_manual(values = vColors[c(2,1)]) +
  scale_color_manual(values = vColors[c(2,1)]) +
  guides(fill=guide_legend(title="Response")) +
  labs(title = "3. Outcome Old") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position = "none")

dfTrain_Data$X3 = factor(dfTrain_Data$X3,levels(dfTrain_Data$X3)[c(2,1)])

p5 =
  ggplot(data = dfTrain_Data) +
  geom_mosaic(aes(x = product(X3), fill= y), na.rm = TRUE) + 
  scale_fill_manual(values = vColors[c(2,1)]) +
  scale_color_manual(values = vColors[c(2,1)]) +
  guides(fill=guide_legend(title="Response")) +
  labs(title = "4. X3") + theme(legend.position = "none")

grid.arrange(p1,p3,p4,p5,p2, layout_matrix = matrix(c(1,3,5,2,4,5),3,2))

# To reset factor level order again:
dfTrain_Data <- read.csv("train.csv")
dfTrain_Data <- dfTrain_Data %>% mutate(y = factor(y), X1 = factor(X1), X2 = factor(X2), X3 = factor(X3))

# Correlation Plot

dfCor.pearson = dfTrain_Data %>% mutate(y = as.integer(y),X1 = as.integer(X1),X2 = as.integer(X2),X3 = as.integer(X3)) %>% select_if(is.numeric) %>% cor()

dfCor.spearman = dfTrain_Data %>% mutate(y = as.integer(y),X1 = as.integer(X1),X2 = as.integer(X2),X3 = as.integer(X3)) %>% select_if(is.numeric) %>% cor(method ="spearman")


p.cor1 =
ggcorr(dfCor.pearson, label = TRUE , 
       nbreaks = 10, 
       label_alpha = 1, 
       size = 3, 
       label_color = "black", 
       layout.exp = 1, 
       hjust = 0.8,  
       angle = -0, 
       low = vColors[1],
       high = vColors[2]) + ggtitle("Pearson Correlation")

p.cor2 =
  ggcorr(dfCor.spearman, label = TRUE , 
         nbreaks = 10, 
         label_alpha = 1, 
         size = 3, 
         label_color = "black", 
         layout.exp = 1, 
         hjust = 0.8,  
         angle = -0, 
         low = vColors[1],
         high = vColors[2])  + ggtitle("Spearman Correlation")

grid.arrange(p.cor1,p.cor2,nrow = 1)

# Distribution comparision of Training/Testing Dataset

dfTest_Data <- read.csv("test.csv")

dfTest_Data <- dfTest_Data %>% mutate(X1 = factor(dfTest_Data$X1), X2 = factor(dfTest_Data$X2), X3 = factor(dfTest_Data$X3))

dfTrain_Data_Indicated = cbind("Train",dfTrain_Data) %>% dplyr::select(-c("y"))

colnames(dfTrain_Data_Indicated)[1] <- "Dataset"

dfTest_Data_Indicated = cbind("Test",dfTest_Data) %>% dplyr::select(-c("ID"))

colnames(dfTest_Data_Indicated)[1] <- "Dataset"

dfCombined_Data = rbind(dfTrain_Data_Indicated,dfTest_Data_Indicated)

# Comparision Continuous Feature distribution by Boxplots

vCol_names_3 = dfCombined_Data %>% select_if(is.numeric) %>% colnames()

lPlot_list_3 = list()


for(i in 1:length(vCol_names_3)){
  
  data = dfCombined_Data %>% dplyr::select(Dataset,vCol_names_3[i])  
  
  colnames(data) = c("X1", "X2")
  
  lPlot_list_3[[i]] <- local({
    i <- i
    p1 <- ggplot(data, aes(x = X1, y = X2)) + geom_boxplot(color = vColors[1:2], fill = vColors[1:2], alpha = 0.4) + 
      xlab("Dataset") + ylab(paste(vCol_names_3[i])) +
      labs(title = paste0(i,". ",vCol_names_3[i]))
  })
  
}

do.call("grid.arrange", c(lPlot_list_3, ncol=4)) 


# Comparision Discrete Feature distribution by Barcharts

vCol_names_4 = dfCombined_Data %>% select_if(is.factor) %>% colnames() 

vCol_names_4 = c(vCol_names_4[2:9],vCol_names_4[1])

lPlot_list_4 = list()


for(i in 1:(length(vCol_names_4)-1)){
  
  data = dfCombined_Data %>% dplyr::select("Dataset",vCol_names_4[i])  
  
  colnames(data) = c("X1", "X2")
  
  lPlot_list_4[[i]] <- local({
    i <- i
    p1 <- ggplot(data, aes(y = X2, fill = X1)) + geom_bar(position="fill", alpha = 0.8) + 
      ylab(paste(vCol_names_4[i])) +
      xlab("Percent") +
      scale_fill_manual(values = vColors[c(1,2)]) +
      guides(fill=guide_legend(title="Dataset")) +
      labs(title = paste0(i,". ",vCol_names_4[i])) + theme(legend.position = "none")
  })
  
  
}

lPlot_list_4[[8]] = lPlot_list_4[[8]] + theme(legend.position = "right")
do.call("grid.arrange", c(lPlot_list_4, ncol=4)) 

