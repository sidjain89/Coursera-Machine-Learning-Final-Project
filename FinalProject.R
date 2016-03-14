library(randomForest)
library(caret)
library(ggplot2)
library(rpart)
library(RCurl)


# Section 0: Data Pull

#trainPath <- "C:/Users/Siddharth/Documents/Coursera R/data/pml-training.csv"
#validPath <- "C:/Users/Siddharth/Documents/Coursera R/data/pml-testing.csv"

# URL of the training and testing data
train.url ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
valid.url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
# file names
train.name = "./data/pml-training.csv"
valid.name = "./data/pml-testing.csv"



# if directory does not exist, create new
if (!file.exists("./data")) {
  dir.create("./data")
}
# if files does not exist, download the files
if (!file.exists(train.name)) {
  download.file(getURL(train.url), destfile=train.name, method="curl")
}
if (!file.exists(valid.name)) {
  download.file(getURL(valid.url), destfile=valid.name, method="curl")
}
# load the CSV files as data.frame 
dat = read.csv("./data/pml-training.csv")
valid = read.csv("./data/pml-testing.csv")


dat <- read.csv(trainPath)
valid <- read.csv(validPath)

###############################################################################################################

# summary(dat)

# Section 1: Data Cleaning

#####################################################
# This function will prune data for columns which   #
# are not useful for modelling. Removed time stamps #
# , columns with 19000+ missing values, iDs etc.    #        
#####################################################
PruneColumns <- function(dat)
{
  dat <- subset(dat, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp,
                                      max_yaw_belt, min_yaw_belt, yaw_belt, pitch_arm, pitch_dumbbell,
                                      new_window, kurtosis_roll_belt, kurtosis_picth_belt, kurtosis_yaw_belt,
                                      skewness_roll_belt, skewness_roll_belt.1, skewness_yaw_belt, max_roll_belt,
                                      max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_roll_belt,
                                      amplitude_pitch_belt, amplitude_yaw_belt, var_total_accel_belt, avg_roll_belt, 
                                      stddev_roll_belt, var_roll_belt, avg_pitch_belt,
                                      stddev_pitch_belt, var_pitch_belt, avg_yaw_belt,      
                                      stddev_yaw_belt, var_yaw_belt,var_accel_arm,
                                      avg_roll_arm, stddev_roll_arm, var_roll_arm,       
                                      avg_pitch_arm, stddev_pitch_arm, var_pitch_arm,
                                      avg_yaw_arm, stddev_yaw_arm, var_yaw_arm,
                                      kurtosis_roll_arm, kurtosis_picth_arm, kurtosis_yaw_arm,
                                      skewness_roll_arm, skewness_pitch_arm, skewness_yaw_arm,
                                      max_roll_arm, max_picth_arm, max_yaw_arm, min_roll_arm,
                                      min_pitch_arm, min_yaw_arm, amplitude_roll_arm,
                                      amplitude_pitch_arm, amplitude_yaw_arm,
                                      kurtosis_roll_dumbbell, kurtosis_picth_dumbbell, kurtosis_yaw_dumbbell, 
                                      skewness_roll_dumbbell, skewness_pitch_dumbbell,
                                      skewness_yaw_dumbbell, max_roll_dumbbell, max_picth_dumbbell,
                                      max_yaw_dumbbell, min_roll_dumbbell, min_pitch_dumbbell,
                                      min_yaw_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell,
                                      amplitude_yaw_dumbbell, var_accel_dumbbell, avg_roll_dumbbell,
                                      stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, 
                                      stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell,
                                      stddev_yaw_dumbbell, var_yaw_dumbbell, yaw_forearm,
                                      kurtosis_roll_forearm, kurtosis_picth_forearm,
                                      kurtosis_yaw_forearm, skewness_roll_forearm,
                                      skewness_pitch_forearm, skewness_yaw_forearm, max_roll_forearm,
                                      max_picth_forearm, max_yaw_forearm, min_roll_forearm,
                                      min_pitch_forearm, min_yaw_forearm, amplitude_roll_forearm,
                                      amplitude_pitch_forearm, amplitude_yaw_forearm,
                                      total_accel_forearm, var_accel_forearm, avg_roll_forearm,
                                      stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm,
                                      stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm,
                                      stddev_yaw_forearm, var_yaw_forearm))
  
  return(dat)
}


dat <- PruneColumns(dat)
valid <- PruneColumns(valid)

#write.csv(dat, file = "D:/courseraML_orig.csv")

# Section 2: Data Transformation
# for all numeric variables do a z- transformation.
# this will normalize values and prune bias

###########################################
# This function will spit out the Z-Score #
###########################################
ZTransform <- function(x)
{
  mn <- mean(x)
  #print(mn)
  stdDev <- sd(x, na.rm = TRUE)
  #print(stdDev)
  
  for(i in 1:length(x))
  {
    if(is.nan(x[i]))
    {
      x[i] <- mn
    }
    x[i] <- (x[i] - mn) / stdDev
  }
  return(x)
}

lastCol <- ncol(dat)
numeric_cols <- colnames(dat)[-49]

####### apply the z transformation ########################
for(column in numeric_cols)
{
  #newName <- paste("z_", column, sep = "")
  dat[,column] <- ZTransform(dat[,column])
  valid[, column] <- ZTransform(valid[,column])
}

#write.csv(dat, file = "D:/courseraML.csv")

inTrain <- createDataPartition(dat$classe, p = 0.7, list = FALSE)
trainset <- dat[inTrain,]
testset <- dat[-inTrain,]

# summary(dat)
# 
# warnings()

rforest <- randomForest(classe ~., data = trainset, nodesize = 10, ntree = 50)
rforest


# get accuracy for each class
nums <- rforest$confusion[,6]
nums <- lapply(nums, as.numeric)
nums <- sapply(nums, function(x) (1-x)*100)
nums

#get the most importent variables
varImpPlot(rforest)
varImp(rforest)


# apply the model to the test set
# create a confusion matrix
pred_rf <- predict(rforest, newdata = testset)
#table(pred_rf, testset$classe)
confusionMatrix(pred_rf, factor(testset$classe))

# apply the random forest model to the validation set
pred_valid <- predict(rforest, newdata = valid)
pred_valid



