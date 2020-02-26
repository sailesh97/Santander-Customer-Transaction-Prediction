##################################################################

#                       Load the libraries                       #

##################################################################

options(warn=-1)
library(tidyverse) 
library(ggplot2)
library(randomForest)
library(ROSE)
library(factoextra)
library(ggsignif)
library(pROC)
library(MLeval)
library(glmnet)
library(caret)
library(e1071)
library(DMwR)
library(caret)

##################################################################

#                       Load the dataset                        #

##################################################################

df_santander<-read.csv('C:/Users/Acesocloud/Downloads/Kaggle/Santander Customer Transaction Prediction/Notebooks/train.csv')
df_santander_test<-read.csv('C:/Users/Acesocloud/Downloads/Kaggle/Santander Customer Transaction Prediction/Notebooks/train.csv')


#Shape of train data
dim(df_santander)

#Shape of test data
dim(df_santander_test)

# # Explororatory Data Analysis

##Showing 1st few rows of our dataset:
head(df_santander,5)

str(df_santander)

##################################################################

#                   TARGET CLASS IMBALANCE                       #

##################################################################


##Count of categories of the target variable:
table(df_santander$target)

##Percentage of each category of the target variable:
table(df_santander$target)/length(df_santander$target)*100


# ### Data Visualization

plot2<-pie(table(df_santander$target), labels = paste(round(prop.table(table(df_santander$target))*100), "%", sep = ""), 
           radius =1,col=rainbow(2), main = "Target Class Count")
legend("topright",fill = rainbow(2), legend = c("0","1"), cex = 2)

ggplot(df_santander,aes(target))+theme_bw()+geom_bar(stat='count',fill=rainbow(2))


##################################################################

#                   MISSING VALUE ANALYSIS                       #

##################################################################


## No. of rows having missing values in train data:
missing_val<-data.frame(missing_val=apply(df_santander,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val



## No. of rows having missing values in test data:
missing_val<-data.frame(missing_val=apply(df_santander_test,2,function(x){sum(is.na(x))}))
missing_val<-sum(missing_val)
missing_val



##################################################################

#                       Outlier Analysis                         #

##################################################################
### CAN NOT PERFORM BECAUSE OF TARGET CLASS IMBALANCE PROBLEM



##################################################################

#                       Distribution Plots                       #

##################################################################


for (var in names(df_santander)[c(3:102)]){
  target<-df_santander$target
  xlabel <- paste("Distribution of",var)
  plot<-ggplot(df_santander, aes(x=df_santander[[var]],fill=target)) +
    geom_density(kernel='gaussian')+ggtitle(xlabel) +theme_classic()+xlab(var)
  print(plot)
}

for (var in names(df_santander)[c(103:202)]){
  target<-df_santander$target
  xlabel <- paste("Distribution of",var)
  plot<-ggplot(df_santander, aes(x=df_santander[[var]],fill=target)) +
    geom_density(kernel='gaussian')+ggtitle(xlabel) +theme_classic()+xlab(var)
  print(plot)
}



# ### Check for duplicate rows

s<-duplicated(df_santander) | duplicated(df_santander, fromLast = TRUE)

table(s)

s2<-duplicated(df_santander_test) | duplicated(df_santander_test, fromLast = TRUE)

table(s2)


##################################################################

#                     Correlation Analysis                      #

##################################################################


# #### Correlation Between Train data

train_correlations<-cor(df_santander[,c(2:202)])
summary(train_correlations)

# #### Correlation between test data

test_correlations<-cor(df_santander_test[,c(2:201)])
summary(train_correlations)

##################################################################

#                       Feature Importance                       #

##################################################################


X_index<-sample(1:nrow(df_santander),0.75*nrow(df_santander))
X<-df_santander[X_index,]
y<-df_santander[-X_index,]

X$target<-as.factor(X$target)
y$target <- as.factor(y$target)

set.seed(2732)
#convert to int to factor
X$target<-as.factor(X$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)

#fitting the ranndom forest
library('randomForest')
rf<-randomForest(target~.,X[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)

VarImp<-importance(rf,type=2)

VarImp[order(-VarImp[,1]), ]

set.seed(689)
train.index<-sample(1:nrow(df_santander),0.7*nrow(df_santander))

### Normal X & y

#train data
X<-df_santander[train.index,]
#validation data
y<-df_santander[-train.index,]

X<-X[,-c(1)]
y<-y[,-c(1)]

#target classes in train data
(table(X$target)/length(X$target))*100
#target classes in validation data
(table(y$target)/length(y$target))*100

# Oversampled Dataset

set.seed(699)
#### Oversample X & y
library('ROSE')

X_rose <- ROSE(target~., data =X[,],seed=32)$data
y_rose <- ROSE(target~., data =y[,],seed=42)$data
X_rose$target<-as.factor(X_rose$target)
y_rose$target <- as.factor(y_rose$target)

rm("df_santander","X_train")
rm("y_train","X_test","y_test","X_train_rs","y_train_rs","X_test_rs","y_test_rs","x_pc","y_pc")
rm(train.index)
rm(X_index)
rm(test_correlations,train_correlations)
rm(df_santander,df_santander_test,rf,tuneGrid,VarImp,missing_val,mtry)
rm(plot2)

# ### PCA
pc <- prcomp((X[,1:200]), scale=TRUE)
summary(pc)

## SCATTER PLOT SHOWING SEPARABILITY OF TARGET CLASS
plot(pc$x[,2],pc$x[,3],col=factor(X$target))

## SCREE PLOT
fviz_eig(pc)

rm(pc)

###########################LOGISTIC REGRESIION#################
mod_fit <- train(target ~.,  data=X, method="glm", family="binomial")
pred = predict(mod_fit, newdata=y,type='raw')
y_pred_num <- ifelse(pred > 0.5, 1, 0)
roc(y[,'target'],pred,plot=TRUE,legacy.axes=TRUE)
y_pred_num <- as.factor(y_pred_num)
confusionMatrix(data=y_pred_num, as.factor(y[,'target']))
PRcurve(pred,y[,'target'])

###########################LOGISTIC REGRESIION ON ROSR DATASET#################

mod_fit <- train(target ~.,  data=X_rose, method="glm", family="binomial")
pred = predict(mod_fit, newdata=y_rose,type="raw")
y_pred_num <- ifelse(pred > 0.5, 1, 0)
roc((y_rose[,'target']),as.numeric(pred),plot=TRUE,legacy.axes=TRUE)
cm<-confusionMatrix(data=pred, as.factor(y_rose[,'target']))
cm
PRcurve(as.numeric(pred),(y_rose[,'target']))

##################DECISION TREE##############################
mod_fit <- train(target ~.,  data=X, method="rpart")
pred = predict(mod_fit, newdata=y,type="raw")
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred_num <- as.factor(y_pred_num)
confusionMatrix(data=y_pred_num, as.factor(y[,'target']))

library(PRROC)


fg <- pred[y$target == 1]
bg <- pred[y$target == 0]

# ROC Curve    
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)
##################DECISION TREE on ROSE Dataset##############################
mod_fit_dt_rose <- train(target ~.,  data=X_rose, method="rpart")
pred_dt_rose = predict(mod_fit_dt_rose, newdata=y_rose,type='raw')
y_pred_num_dt_rose <- ifelse(pred_dt_rose > 0.5, 1, 0)
y_pred_num_dt_rose <- as.factor(y_pred_num_dt_rose)
confusionMatrix(data=pred_dt_rose, (y_rose[,'target']))
fg_rs <- pred_dt_rose[y_rose$target == 1]
bg_rs <- pred_dt_rose[y_rose$target == 0]

# ROC Curve    
roc_rs <- roc.curve(scores.class0 = fg_rs, scores.class1 = bg_rs, curve = T)
plot(roc_rs)

# PR Curve
pr_rs <- pr.curve(scores.class0 = fg_rs, scores.class1 = bg_rs, curve = T)
plot(pr_rs)


##############NAIVE BAYES##########################
model_fit <- train(X[,2:201],X$target,'nb',trControl=trainControl(method='cv',number=10))
pred = predict(model_fit, newdata=y,type='raw')
y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred_num <- as.factor(y_pred_num)
confusionMatrix(data=pred, as.factor(y[,'target']))
fg <- pred[y$target == 1]
bg <- pred[y$target == 0]

# ROC Curve    
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

# PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)

#####################NAIVE BAYES ON OVERSAMPLED DATASET##################
model_fit2 <- train(X_rose[,2:201],X_rose$target,'nb',trControl=trainControl(method='cv',number=10))
pred2 = predict(model_fit2, newdata=y_rose,type='raw')
y_pred_num2 <- ifelse(pred2 > 0.5, 1, 0)
y_pred_num2 <- as.factor(y_pred_num2)
confusionMatrix(data=pred2, as.factor(y_rose[,'target']))
fg1 <- pred2[y_rose$target == 1]
bg1 <- pred2[y_rose$target == 0]

# ROC Curve    
roc1 <- roc.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(roc)
# PR Curve
pr2 <- pr.curve(scores.class0 = fg1, scores.class1 = bg1, curve = T)
plot(pr2)