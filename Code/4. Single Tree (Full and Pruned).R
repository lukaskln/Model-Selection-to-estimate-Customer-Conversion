source(paste0(getwd(),"/ML_R_Code_Group_KLC/1. Descriptive and Correlation Analysis.R")) # Load Data and libraries from EDA part, change path if needed
source(paste0(getwd(),"/ML_R_Code_Group_KLC/3. Feature Engineering.R")) # Transformed Dataset

# Single Tree

set.seed(123)
model.tree = rpart(y ~., 
                   data = dfTrain_Data, 
                   method = "class",
                   parms = list(split = "gini"), 
                   cp = 0)

model.tree$cptable

visTree(model.tree, 
        main = "Single Tree fully grown", 
        edgesFontSize = 13, 
        nodesFontSize = 13, 
        width = "100%", height = "800px",
        colorVar = vColors[1:5], 
        colorY = c("red","green"), 
        legend = TRUE) # Html file


# Pruned Tree

set.seed(123)
model.tree.pruned = prune.rpart(model.tree, cp = 0.01)

model.tree.pruned$cptable

visTree(model.tree.pruned, 
        main = "Single Tree Pruned", 
        edgesFontSize = 13, 
        nodesFontSize = 13, 
        width = "100%", height = "800px",
        colorVar = vColors[3], colorY = vColors[1:2], colorEdges = vColors[4], 
        legend = FALSE, 
        direction = "LR")
