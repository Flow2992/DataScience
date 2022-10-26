# Data Science: Capstone - Choose Your Own Project
# Dataset: Stroke Data
# Data set consists of data from people who have had a stroke and those who have not. 
# In addition to the information on whether a stroke has occurred, other information such as health data is available. 
# The goal of this project is to predict whether a patient has had a stroke or not.
# The goal is to predict whether a patien has had a stroke or not. 
# For this purpose, the data set is first analyzed and then a model is developed.

##########################################################################################################################################################
# Package preparation
##########################################################################################################################################################

# Check if required packages are installed, if not, install required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("carnet", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(imbalance)) install.packages("imbalance", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(margins)) install.packages("margins", repos = "http://cran.us.r-project.org")
if(!require(yardstick)) install.packages("yardstick", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")
if(!require(evaluate)) install.packages("evaluate", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

# Load required packages
library(tidyverse)
library(caret)
library(data.table)
library(naniar)
library(caret)
library(caTools)
library(corrplot)
library(randomForest)
library(imbalance)
library(MASS)
library(ROSE)
library(broom)
library(margins)
library(yardstick)
library(ROCR)
library(glmnet)
library(ranger)
library(evaluate)
library(kernlab)
library(xgboost)

##########################################################################################################################################################
# Load Data
##########################################################################################################################################################

# Dataset is online available to download: 
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?datasetId=1120859&language=R 

data <- read.csv("https://raw.githubusercontent.com/Flow2992/DataScience/main/healthcare-dataset-stroke-data.csv") # Load Dataset from github
data <- as.data.table(data)

##########################################################################################################################################################
# Data preparation
##########################################################################################################################################################

# View rough data structure
str(data)
summary (data)

# Delete the ID column, since it is not needed.
data = subset(data, select = -c(id))

# Check if data is missing in a variable
miss_scan_count(data = data, search = list("N/A", "Unknown")) # Data are missing in the variables "bmi" and "smoking_status"

# Check which values the individual variables have.
# The variables "age", "avg_glucose_level" and "bmi" are not checked, because they contain to many different values
table(data$gender)
table(data$hypertension)
table(data$heart_disease)
table(data$ever_married)
table(data$work_type)
table(data$Residence_type)
table(data$smoking_status)
table(data$stroke)

# There is one record with the gender "other", this will be removed
data <- data %>% filter(data$gender!='Other')

table(data$gender)

# Handling the missing values
# Replace missing bmi values with mean values by gender since men have a higher bmi than women 
suppressWarnings(data$bmi <- as.numeric(as.character(data$bmi))) # transform bmi column to numeric, because of the missing values it was not numeric

gender_mean_bmi <- data %>% group_by(gender) %>% summarise(bmi = mean(bmi, na.rm = TRUE)) # calculate gender bmi
gender_mean_bmi

data[gender == 'Female' & is.na(data$bmi), bmi := gender_mean_bmi[1, 'bmi']] # replace missing female bmi values calculated mean bmi
data[gender == 'Male'   & is.na(data$bmi), bmi := gender_mean_bmi[2, 'bmi']] # replace missing male bmi values calculated mean bmi

# Recheck for missing data
miss_scan_count(data = data, search = list("N/A", "Unknown"))

table(data$smoking_status) # smoking status contains a lot of Values "Unknown".

# Sicne the smoking_status contains a lot of values of the status "Unknown". This status is not suitable for further analysis, therefore the unknown 
# values are replaced. In order not to change the data too much, the status "Unknown" is replaced according to the distribution of the other values. 
# The distribution of the other statuses is calculated and the status "Unknown" is replaced according to the distribution.

table(data$smoking_status) # Check smoking values distibution

s_dist1 <- ggplot(data, aes(x = smoking_status)) + geom_bar()  # safe plot distribution before replacing the status "Unknown"

FS <- 885 / (3566) # Calculate probability for status formerly smoked
NS <- 1892 / (3566) # Calculate probability for status never smoked
S <- 789 / (3566) # Calculate probability for status smokes

# Replace missing values on the basis of the calculated probabilities
data$prob <- runif(nrow(data))
data <- data %>% mutate(Proba = ifelse(prob <= FS, "formerly smoked", ifelse(prob <= (FS+NS), "never smoked", ifelse(prob <= 1, "smokes", "Check"))))
data <- data %>% mutate(smoking_status = ifelse(smoking_status == "Unknown", Proba, smoking_status))

table(data$smoking_status) # ReCheck smoking values distibution

# Delete supporting columns for replacing the status
data = subset(data, select = -c(prob))
data = subset(data, select = -c(Proba))

miss_scan_count(data = data, search = list("N/A", "Unknown")) # recheck dataset for missing values

s_dist2 <- ggplot(data, aes(x = smoking_status)) + geom_bar() # safe plot distibution after replacing the status "Unknown"

# Visual Comparison of Distibution of smoking values before and after replacing "Unknown" values
s_dist1
s_dist2 # overall distibution after replacing the "Unknown" status didn't change much, so the replacment worked well

##########################################################################################################################################################
# Data exploration
##########################################################################################################################################################

# Visual inspection of the data distribution to learn more about the data
ggplot(data, aes(x = gender)) + geom_bar() # Gender distribution
ggplot(data, aes(x = smoking_status)) + geom_bar() # smoking_status distribution
ggplot(data, aes(x = heart_disease)) + geom_bar() # heart_disease distribution
ggplot(data, aes(x = ever_married)) + geom_bar() # ever_married distribution
ggplot(data, aes(x = Residence_type)) + geom_bar() # Residence_type distribution
ggplot(data, aes(x = work_type)) + geom_bar() # work_type distribution
ggplot(data, aes(x = stroke)) + geom_bar() # stroke distribution
ggplot(data, aes(x = age)) + geom_histogram(bins=30) # age distribution
ggplot(data, aes(x = bmi)) + geom_histogram(bins=30) # bmi distribution
ggplot(data, aes(x = avg_glucose_level)) + geom_histogram(bins=30) # avg_glucose_level distribution

# Visual inspection of the data distribution of variables with many expressions in combination with the 
# stroke data to learn more about the data and potential correlations.

ggplot(data, aes(x = stroke, y = age, group = stroke)) +geom_boxplot() # combination of age and stroke -> age seems to have some sort of impact on stroke
ggplot(data, aes(x = stroke, y = bmi, group = stroke)) +geom_boxplot() # combination of bmi and stroke -> no clear visual correlation
ggplot(data, aes(x = stroke, y = avg_glucose_level, group = stroke)) +geom_boxplot() # combination of avg_glucose_level and stroke -> correlation possible

# For further analysis, some text values of the variables have to be converted into numerical values
# gender
data$gender[data$gender == "Male"] <- 1 # Male --> 1
data$gender[data$gender == "Female"] <- 0 # Female --> 0

# residence typ
data$Residence_type[data$Residence_type == "Urban"] <- 1 # Urlan --> 1
data$Residence_type[data$Residence_type == "Rural"] <- 0 # Rural --> 0

# ever married
data$ever_married[data$ever_married == "Yes"] <- 1 # yes --> 1
data$ever_married[data$ever_married == "No"] <- 0 # no --> 0

# check dataset after all transformations
str(data) # just transformed variables are not yet numeric
head(data)

# For further analysis, the variables just transformed are converted into numerical values
suppressWarnings(data$gender <- as.numeric(as.character(data$gender)))
suppressWarnings(data$Residence_type <- as.numeric(as.character(data$Residence_type)))
suppressWarnings(data$ever_married <- as.numeric(as.character(data$ever_married)))

# Further visual analysis of the data in connection with stoke, which has not yet been visually examined for correlation
# The goal is to find out which characteristic of the variables could have an influence on stroke 

data$stroke_1 = ifelse(data$stroke == 1, 'stroke', 'no stroke') # create new stroke variable
data$stroke_1 = factor(data$stroke_1) # transform new variable as factor

ggplot(data, aes(x = gender, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of gender and stroke -> no clear visual impact
ggplot(data, aes(x = smoking_status, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of smoke status and stroke -> no clear visual impact
ggplot(data, aes(x = work_type, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of work type and stroke -> work types "self-employed", "Govt_job" and "Private" could have impact
ggplot(data, aes(x = ever_married, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of bmi and stroke -> marriage could have impact
ggplot(data, aes(x = heart_disease, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of bmi and stroke -> heart disease could have impact
ggplot(data, aes(x = hypertension, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of bmi and stroke -> hypertension could have impact
ggplot(data, aes(x = Residence_type, fill = stroke_1))+ geom_bar(position = "fill" , alpha = 0.3) # combination of bmi and stroke -> no clear visual impact

# Remove the just added support column stroke_1
data = subset(data, select = -c(stroke_1))

# The final step of data exploration is to examine the correlation of all data 
str(data) # For this analysis, all data must be available in numerical form, we see that there is data, that is not yet numerical

data$hypertension = as.numeric(as.character(data$hypertension)) # transform to numerical data
data$heart_disease = as.numeric(as.character(data$heart_disease)) # transform to numerical data

data$work_type = str_replace_all(data$work_type, c("Never_worked"="0","children"="1", "Private"="2", "Self-employed"="3", "Govt_job"="4")) # replace text with numbers
data$work_type = as.numeric(data$work_type) # transform to numerical data

data$smoking_status = str_replace_all(data$smoking_status, c("never smoked"="0","formerly smoked"="1", "smokes"="2")) # replace text with numbers
data$smoking_status = as.numeric(data$smoking_status) # transform to numerical data

data$stroke = as.numeric(as.character(data$stroke))

str(data) # recheck if all date is numerical now

# Examine the data set for correlation
correlation <- cor(data)
corrplot(correlation, type = "upper", order = "hclust", tl.srt = 50) # one can see that it works. for example, there is a high correlation between married status and age
# As in the previous single variable analysis, there is no particularly strong correlation between stoke and any particular variable. 
# However, it can be seen that age, hypertension, glucose lever heart disease, and marriage status may show a correlation.

##########################################################################################################################################################
# Modeling
##########################################################################################################################################################

# Logistic Regression
# Logistic regression is used to make predictions about categorical variables, whereas linear regression is used to make predictions about a continuous variable.
# The model should predict whether a stroke occurs or not. 
# Since the result of the prediction can only take two forms, stroke or no stroke (categorical variable), a logistic regression is used.

# Split data in training and test data with 80% training data
set.seed(5)
test_index <- createDataPartition(data$stroke, times = 1, p = 0.8, list = FALSE)
test <- data[-test_index, ]
train <- data[test_index, ]

dim(train) 
dim(test)

# Check the probability of stoke in the two datasets to ensure that the split is usable
prop.table(table(train$stroke))
prop.table(table(test$stroke)) # The probability is approximately the same, so the datasets are appropriate

# Create Generalized linear model with family = binomial
glm_regression <- glm(stroke~., data=train, family=binomial)

summary(glm_regression) # check the model to see which influencing factors the model sees. The influencing factors found 
# by the model are consistent with those from the previous corellation analysis

# Test the model using the test set
prediction <- predict(glm_regression, test, type="response")

pred_test <- ifelse(prediction >0.5,1,0) # if prediction over 0.5 than stroke prediction
fourfoldplot(table(Prediction = pred_test, Real = test$stroke), conf.level = 0, margin = 1) # check confusion matrix -- > High number of false negative, few true positives
print(1-mean(pred_test != test$stroke)) # High Accuracy

# The threshold from which a stoke is predicted is apparently too high, therefore test with a different threshold

pred_test <- ifelse(prediction >0.4,1,0)
fourfoldplot(table(Prediction = pred_test, Real = test$stroke), conf.level = 0, margin = 1) # check confusion matrix -- > no change to previous setting
print(1-mean(pred_test != test$stroke)) # same high Accuracy as before

pred_test <- ifelse(prediction >0.3,1,0)
fourfoldplot(table(Prediction = pred_test, Real = test$stroke), conf.level = 0, margin = 1) # check confusion matrix -- > more true positives, but also more false positives
print(1-mean(pred_test != test$stroke)) # Accuracy just litte changed

pred_test <- ifelse(prediction >0.2,1,0)
fourfoldplot(table(Prediction = pred_test, Real = test$stroke), conf.level = 0, margin = 1) # check confusion matrix -- > more true positives, fewer false negatives
print(1-mean(pred_test != test$stroke)) # Accuracy lower than before, but still good

pred_test <- ifelse(prediction >0.1,1,0)
fourfoldplot(table(Prediction = pred_test, Real = test$stroke), conf.level = 0, margin = 1) # highest number of true positves and lowest number of false negatives.
print(1-mean(pred_test != test$stroke)) # Accuracy lower than before, but still ok
# With this setting the number of false positives is the highest, but the number of false negatives is the lowest. 
# In this case, false positives are less bad than false negatives, so we stick with these settings. 
# For the fact that the number of true positives goes up, the number of false positives also goes up. In this case, however, it is better to predict 
# false positives and have more true positives than to have fewer false positives and fewer true positives.
# At the expense of Accuracy the number of true positives increases
# With these results, the model could be used as a kind of warning system in this case

# Based on the previous model summary and the performed data exploration one can see that age, hypertension, heard disease and glucose levels 
# could have biggest impact out of all variables
# Next step ist to try a models with only these variables an see if the pervious results could be increased

# create new dataset for this test
data1 <- data

# remove not needed columns 
data1 = subset(data1, select = -c(gender, ever_married, work_type, Residence_type, bmi, smoking_status))

# check newly created dataset
str(data1)

# Split data in training and test data with 80% training data
set.seed(5)
test_index <- createDataPartition(data1$stroke, times = 1, p = 0.8, list = FALSE)
test1 <- data1[-test_index, ]
train1 <- data1[test_index, ]

# check Propability for stroke in train and test data
prop.table(table(train1$stroke))
prop.table(table(test1$stroke))

# Create Generalized linear model with family = binomial
glm_regression1 <- glm(stroke~., data=train1, family=binomial)

summary(glm_regression1) # summarize model

# Test the model using the test set
prediction1 <- predict(glm_regression1, test1, type="response")

pred_test1 <- ifelse(prediction1 >0.1,1,0) # use best setting from first model
fourfoldplot(table(Prediction = pred_test1, Real = test1$stroke), conf.level = 0, margin = 1)
print(1-mean(pred_test1 != test1$stroke))

# Both the confusion matix and the accuracy are almost equal to the previous test
# The model could therefore not be improved in this way. 
# But the result show that the variables that were dropped in this case have no real influence on the model.

# Next, the stepwise regression is tested to automaticly perform the previous manual step and try the model with differnet cominations of the variables 
# Dataset from fist model is used, so that the model can chose from all variables  

glm_regression_steps = glm(stroke~., data=train, family = "binomial") %>% stepAIC(trace = TRUE)

prediction2 <- predict(glm_regression_steps, test, type="response") # predict with steps model

pred_test2 <- ifelse(prediction2 >0.1,1,0) # use the best setting from first model
fourfoldplot(table(Prediction = pred_test2, Real = test$stroke), conf.level = 0, margin = 1) # Not much difference from the previous two attempts. The number of false positives drops only minimally
print(1-mean(pred_test2 != test$stroke))

# All attempts to change the combination of variables did not bring much. The results all remain very similar

# Since the number of stokes is very small compared to the number of non-strokes, the last option is to oversample to artificially adjust the number of strokes and non-strokes.
# The dataset that was used for the first model is now used again and oversampling is applied to it

data2 <- ovun.sample(stroke~.,data = data, method = 'over',p = 0.3)$data # the number of strokes will artificially be increased

table(data$stroke) # bevore oversampling
table(data2$stroke) # after oversampling --> number of strokes increased significant

set.seed(5)
test_index <- createDataPartition(data2$stroke, times = 1, p = 0.8, list = FALSE)
test2 <- data2[-test_index, ]
train2 <- data2[test_index, ]

# check if probability for stroke is approximately the same in the train and test dataset
prop.table(table(train2$stroke)) 
prop.table(table(test2$stroke)) # probability is approximately the same and also here you can see the result of the oversampling, 
# because the probability for stoke in this data is much higher than in the previous data.

glm_regression2 <- glm(stroke~., data=train2, family=binomial)

summary(glm_regression2)

prediction3 <- predict(glm_regression2, test2, type="response")

pred_test3 <- ifelse(prediction3 >0.1,1,0) # use the best setting from first model
fourfoldplot(table(Prediction = pred_test3, Real = test2$stroke), conf.level = 0, margin = 1) 
print(1-mean(pred_test3 != test2$stroke)) #  
# Even though it is not really comparable to the previous models without oversampling, the number of false negatives is lower compared to the models without oversampling, what is good from a medical point of view
# The accuracy droped significantly

# Testing the algorithm trained with oversampling with the test data without oversampling
prediction4 <- predict(glm_regression2, test, type="response")

pred_test4 <- ifelse(prediction4 >0.1,1,0) # use the best setting from first model
fourfoldplot(table(Prediction = pred_test4, Real = test$stroke), conf.level = 0, margin = 1) # Not much difference from the previous two attempts. The number of false positives drops only minimally
print(1-mean(pred_test4 != test$stroke))
# In this case, the number of false negatives is very low and the number of true positives is the highest of all those tested.
# From a medical point of view, this model is therefore the best at first glance.
# Unfortunately, the number of false positives is also very high and the accuracy very low.
# So it looks like the first model is still the best so far

# Random Forest
# next, the random forest model is tested, as this model can also be used to predict 
# classifications and is used for input variables without much correlation.

# the train and test dataset from the first model will be used, but the searched variable, stroke, is transformed into a factor
train$stroke <- as.character(train$stroke)
train$stroke <- as.factor(train$stroke)
test$stroke <- as.character(test$stroke)
test$stroke <- as.factor(test$stroke)

# create random forest model
ran_for = randomForest(stroke~., train, importance=TRUE)
summary(ran_for)
ran_for

# test model on test data
rf_predict = predict(ran_for, test)

fourfoldplot(table(Prediction = rf_predict, Real = test$stroke), conf.level = 0, margin = 1) 
print(1-mean(rf_predict != test$stroke)) 
# The model produces very similar results to the very first attempt at logistic regression. 
# There are very few true positives, relatively many false negatives, but very high accuracy
# Since only very few ture positives are detected, the model is not particularly good from a medical point of view.

# Test whether adjusting model variables can improve the model. In this step try to find the optiomal variables to minimize the error rate

NTrees=1:10 # Number of trees (tried with higher nubers, but calculation will take super long)
MTRY=1:10 # Number of variables (tried with higher nubers, but calculation will take super long)
NODE=1:50 # Minimum size of terminal nodes
MinErr=1 
MinNT=0
minNDT=0
minMT=0

for(nt in NTrees){
  for(mt in MTRY){
    for(nd in NODE){
      ran_for2 = randomForest(stroke~., type="classification", train, ntree=nt, mtry=mt, nodesize=nd, importance=TRUE)
      rf_predict2=predict(ran_for2, train)
      Err=mean(rf_predict2 != train$stroke)
      if(Err < MinErr){
        MinErr=Err
        minNT=nt
        minNDT=nd
        minMT=mt
        #print(c("NT=",nt," MT=",mt," NDT=",ndt," minE=",MinError))
      }
    }
  }
}
print(c("NTrees=",minNT," MTRY=",minMT," NODE=",minNDT," MinErr=",MinErr)) # best models variables to minimize the error

# use the best models variables with minimal errror rate and test model
ran_for3 = randomForest(stroke~., type="classification", train, ntree=minNT, mtry=minMT, nodesize=minNDT, importance=TRUE)
summary(ran_for3)
ran_for3

# test optimized model on test data
rf_predict3=predict(ran_for3, test)

fourfoldplot(table(Prediction = rf_predict3, Real = test$stroke), conf.level = 0, margin = 1) 
print(1-mean(rf_predict3 != test$stroke)) 
# With optimal model variables, the model is not performing much better
# The accuracy is still high
# But with still only few true positives the modeel is still not very useful

# next, the model is tested with the oversampling data
table(data2$stroke) # qick look at oversampled data

# transform the searched variable, stroke, into a factor
train2$stroke <- as.character(train2$stroke)
train2$stroke <- as.factor(train2$stroke)
test2$stroke <- as.character(test2$stroke)
test2$stroke <- as.factor(test2$stroke)

# create random forest model with oversampled data
ran_for4 = randomForest(stroke~., train2, importance=TRUE)
summary(ran_for4)
ran_for4 # the error rate has fallen sharply 

# test model on test data
rf_predict4 = predict(ran_for4, test2)

fourfoldplot(table(Prediction = rf_predict4, Real = test2$stroke), conf.level = 0, margin = 1) 
print(1-mean(rf_predict4 != test2$stroke)) # 
# With oversampled data the model performed way better. The number of False positives and false negatives is quit low and the number of true positives is quite high
# The accuracy is also high

# test model on test data withou oversampling
rf_predict5 = predict(ran_for4, test)

fourfoldplot(table(Prediction = rf_predict5, Real = test$stroke), conf.level = 0, margin = 1) 
print(1-mean(rf_predict5 != test$stroke)) # 
# Also with not oversampled test date the model which was trained with oversampled data permores quite good
# The accuracy is also high and the number aof true positives is also high. The number of false positves and fales negatives aus low.
# So far the random forest model, trained with oversampled data permormed the best

# XGBoost
# As last model XGBoost will be used since is very powerful for classification and regression
# Especially in many ML competioions XGBoost has achieved good results

# create train and test data
set.seed(5)
test_index <- createDataPartition(data$stroke, times = 1, p = 0.75, list = FALSE)
test3 <- data[-test_index, drop=FALSE]
train3 <- data[test_index, drop=FALSE]

dim(train3) 
dim(test3)

# transform the searched variable, stroke, into a factor
train$stroke <- as.character(train$stroke)
train$stroke <- as.factor(train$stroke)
test$stroke <- as.character(test$stroke)
test$stroke <- as.factor(test$stroke)

# Since it can be very complex to experiment with the many settings to try this last model just commonly used settings were taken
grid <- expand.grid(nrounds = 3500, max_depth = 7,eta = 0.01, gamma = 0.01, colsample_bytree = 0.75, min_child_weight = 0, subsample = 0.5) # create grid with standard values
Control <- trainControl(method = "cv", number = 5)

# train the model
xgb_model <- caret::train(stroke ~ ., train, method = "xgbTree", tuneLength = 3, tuneGrid = grid, trControl = Control)

xgb_model

# test model with test data
xbg_pred <- predict(xgb_model, newdata = test3) 

fourfoldplot(table(Prediction = xbg_pred, Real = test3$stroke), conf.level = 0, margin = 1) 
print(1-mean(xbg_pred != test3$stroke)) # 

# This model also delivers good results. The tre positives are much highter than on the first model wile the accuracy ist as high as fot the first model.
# Next Steps could be further experimenting wth the model settings


