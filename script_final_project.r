# ----------------------------------------------------------------------------
# Script | final report | Data Science: Capstone course
# Title: Using machine learning to predict thermostable enzymes for
#        second-generation biofuel production based on structural signatures
# Date: 12/25/2019 (yes, I'm finishing this during Christmas)
# Author: Diego Mariano
# ----------------------------------------------------------------------------

# loading the libraries ------------------------------------------------------

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
# for some strange reason, I had to install this package:
# install.packages('e1071', dependencies=TRUE)


# Importing data ---------------------------------------------------------------------
# File type: csv files generated using aCSM tool
# Description: each file contains structural signature vectors of 
#              enzymes used in biofuel production
# t = thermo | nt = not thermo 
t = read.csv("t.csv", header=FALSE)
nt = read.csv("nt.csv", header=FALSE)


# separing train and test 50/50 ------------------------------------------------------
t_train = t[1:50,]
t_test = t[51:100,]

nt_train = nt[1:50,]
nt_test = nt[51:100,]

# Joining train and test ------------------------------------------------------------
train = rbind(t_train,nt_train)
test = rbind(t_test,nt_test)

# Labels ----------------------------------------------------------------------------
l1 = matrix(nrow=50, data="t")
l2 = matrix(nrow=50, data="nt")
labels = rbind(l1,l2)


# TEST 1: guessing the results ----------------------------------------------------
# Description: before constructing the model, I tested my "R habilities" trying to 
#              guess the results. It was (obviously) expected accuracy of ~0.5 :p
y = labels
x = train

# Generating random values using sample function
y_hat = sample(c("t", "nt"), length(y), replace = TRUE)

# overall accuracy
mean(y_hat == y)

# calculating f-measure
F_meas(data=factor(y_hat), reference=factor(y))

# generating confusion matrix
confusionMatrix(data=factor(y_hat), reference=factor(y))

# everything appear ok... I obtained the expected results... 


# ------------------------ the magic starts here ------------------------------------

# declaring variables (I don't know if it is necessary)

acc  = c()  # accuracy
sens = c()  # sensibility
spec = c()  # specificity
fmea = c()  # f-measure

# Separating  "train" dataset: (1) train_final and (2) validation -------------------------------
# Description: I used this to define the best parameters for KNN
#              My objective is to avoid using the test dataset for parameterization.
#              I know that I could implement cross-validation, but due to "technical difficulties"
#              (why is R so complicated?), I decided to implement a simpler strategy

train_final = rbind(train[1:25, ],train[51:75, ])
validation = rbind(train[26:50, ],train[76:100, ]) 

label_train_final = c(labels[1:25, ],labels[51:75, ])
label_validation = c(labels[26:50, ],labels[76:100, ])

# determining the best k value for KNN ----------------------------------------------------
# I tested values from 1 to 50 
for(k in 1:50){
  #print(i)
  
  # training with train_final dataset
  knn_fit = knn3(train_final, factor(label_train_final), k = k)
  
  # checking using validation
  y_hat_knn = predict(knn_fit, validation, type="class")
  
  # and calculating the best metrics
  # confusionMatrix(data=y_hat_knn, reference = factor(label_validation))$overall['Accuracy']
  acc = c(acc,mean(y_hat_knn == label_validation))
  spec = c(spec,specificity(factor(y_hat_knn), factor(label_validation)))
  sens = c(sens,sensitivity(factor(y_hat_knn), factor(label_validation)))
  fmea = c(fmea, F_meas(data=factor(y_hat_knn), reference=factor(label_validation)))
}


# ploting overall accuracy and f-measure ------------------------------------------------------------------
plot(acc, type="n", col="blue", ylim=c(0,1), ylab="Accuracy/F-measure", xlab="Number of Neighbors (k)")
abline(h=c(0.25,0.5,0.75), col="grey", lty="dashed")
lines(acc, type="l", col="blue", lty="solid")
lines(fmea, type="l", col="black", lty="solid" )
abline(h=0.8135, col="red", lty="dashed")
abline(v=c(32), col="red", lty="dashed")

# -------------------------------------------------
# k = 32 appears to be the best k value
# -------------------------------------------------


# I also ploted other metrics, remove the comments if you wanna see
# plot(spec, type="n", col="red", ylim=c(0,1), ylab="Specificity", xlab="Number of Neighbors (k)")
# abline(h=c(0.25,0.5,0.75), col="grey", lty="dashed")
# lines(spec, type="l", col="red", lty="dashed")
# 
# plot(sens, type="n", col="green", ylim=c(0,1), ylab="Sensitivity", xlab="Number of Neighbors (k)")
# abline(h=c(0.25,0.5,0.75), col="grey", lty="dashed")
# lines(sens, type="l", col="green", lty="dashed" )
# 
# plot(fmea, type="n", col="black", ylim=c(0,1), ylab="F-measure", xlab="Number of Neighbors (k)")
# #plot(34,0.82,type="o", col="black", ylim=c(0,1), xlim=c(0,100))
# abline(h=c(0.25,0.5,0.75), col="grey", lty="dashed")
# lines(fmea, type="l", col="black", lty="dashed" )


# --------------------------------- CONSTRUCTING THE MODEL -----------------------------------------
# USING KNN 
# Training: "train" dataset
# Test: "test" dataset

knn_fit = knn3(train, factor(labels), k = 32)
y_hat_knn = predict(knn_fit, test, type="class")

# metrics for evaluation
confusionMatrix(data=y_hat_knn, reference = factor(labels))
F_meas(data=y_hat_knn, reference = factor(labels))


# -------------------------------------------- END ---------------------------------------------