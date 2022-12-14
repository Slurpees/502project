Songs <- read.table("~/Documents/STUDYATNCSU/2022fallsemeter/ST563/FinalProject/YearPredictTrain.txt", header = TRUE, sep = ',')
Songs_test <- read.table("~/Documents/STUDYATNCSU/2022fallsemeter/ST563/FinalProject/YearPredictTest.txt", header = TRUE, sep = ',')
Songs <- Songs[, -91]
Songs_test <- Songs_test[, -91]

X <- Songs[, -91]
# Total variation
TV = sum(apply(X, 2, var))
TV

#Before proceeding, let us check variances of individual predictors.
var<- apply(X, 2, var)
plot(var)

# From the plot we can see that there are some variable having larger
# variances near 15-20 than others.
# So, the Standardizing each predictor is used to avoid the case of imbalance.
# Standardized predictors



# For the response variables
# classes of songs. 
table(Songs$Class)
# The minimum number of each class is 7009
# # So, we resampled rows by different classes.
Songs_1 <- Songs[sample(which(Songs$Class == 'after 2000'),7009), ]
Songs_2 <- Songs[sample(which(Songs$Class == 'between 1980 - 2000'),7009), ]
Songs_3 <- Songs[sample(which(Songs$Class == 'prior to 1980'),7009), ]
Sampled_Songs <- rbind(Songs_1, Songs_2, Songs_3)
# Sampled_Songs <- Songs
#as.factor(Songs$Class)

# Extract only predictors and center them
X1 <- scale(Sampled_Songs[, -91],
            center = TRUE, scale = TRUE)
Sampled_Songs[,1:90] <- X1
dim(X1)

# scale the Test
X <- scale(Songs_test[, -91],
           center = TRUE, scale = TRUE)
Songs_test[,1:90] <- X
dim(X)

# Now we perform PCA of the predictors. 
# PCA
pc_out <- prcomp(X1)
summary(pc_out)
plot(pc_out)
# scarifice 20%  of total variance, then we only need 39 predictors
#add a training set with principal components
# Sampled_Songs <- data.frame(Class = Sampled_Songs$Class, pc_out$x)[,1:46]
# # #transform test into PCA
# Songs_test <- data.frame(Class = Songs_test$Class, predict(pc_out, newdata = Songs_test))[,1:46]

library(MASS)
library(caret)
# LDA
caret_lda <- train(as.factor(Class) ~.,
                   data = Sampled_Songs,
                   method = "lda",
                   trControl = trainControl(method = "CV",
                                            number = 10))
# the test error is 
# Accuracy 0.60(without pca)
caret_lda$results
# details
pred <- predict(caret_lda$finalModel, newdata = Songs_test[, -91])
table(Songs_test$Class)
table(pred$class)
table(Songs_test$Class, pred$class)
confusionMatrix(data = pred$class, reference = Songs_test$Class, mode = "prec_recall")

# QDA

caret_qda <- train(as.factor(Class) ~.,
                   data = Sampled_Songs,
                   method = "qda",
                   trControl = trainControl(method = "CV",
                                            number = 10))
# the test error is 
# Accuracy 0.447(without pca) 
caret_qda$results
# details
pred <- predict(caret_qda$finalModel, newdata = Songs_test[, -91])
table(Songs_test$Class)
table(pred$class)
table(Songs_test$Class, pred$class)
confusionMatrix(data = pred$class, reference = Songs_test$Class, mode = "prec_recall")


# Native Bayesian Classifier
# library(klaR)
# nb_Songs <- NaiveBayes(as.factor(Class) ~ .,
#                       data = Sampled_Songs,
#                       usekernel = TRUE)
# pred <- predict(nb_Songs, newdata = Songs_test[1:10, -91])
grid <- data.frame(fL=c(0,0.3, 0.5, 0.7, 1.0), usekernel = TRUE, adjust=c(0,0.3, 0.5, 0.7, 1.0))
caret_nb <- train(as.factor(Class) ~.,
                  data = Sampled_Songs,
                  method = "nb",
                  tuneGrid=grid,
                  trControl = trainControl(method = "CV",
                                           number = 10)
)
# Accuracy is 0.455(kernel = True)
caret_nb$results
# details
pred <- predict(caret_nb$finalModel, newdata = Songs_test[, -91])
table(Songs_test$Class)
table(pred$class)
table(Songs_test$Class, pred$class)
confusionMatrix(data = pred$class, reference = Songs_test$Class, mode = "prec_recall")

# multinom logistic regression
tuneGrid_mnl <- expand.grid(decay = seq(0, 1, by = 0.1))
caret_multinom <- train(as.factor(Class) ~.,
                        data = Sampled_Songs,
                        method = "multinom",
                        tuneGrid = tuneGrid_mnl,
                        trControl = trainControl(method = "CV",
                                                 number = 10)
)
# multinom 
# Accuracy is 60%
caret_multinom

# details
pred <- predict(caret_multinom$finalModel, newdata = Songs_test[, -91])
table(Songs_test$Class)
table(pred)
table(Songs_test$Class, pred)
confusionMatrix(data = pred, reference = Songs_test$Class, mode = "prec_recall")

# KNN
# k = 67, accuracy is 0.5526 (after pca)
caret_knn <- train(as.factor(Class) ~.,
                   data = Sampled_Songs,
                   method = "knn",
                   trControl = trainControl(method = "CV",
                                            number = 10),
                   tuneGrid = expand.grid(k = seq(1, 101, by = 2))
                                          )
# details
caret_knn
pred <- predict(caret_knn, newdata = Songs_test[, -91])
table(Songs_test$Class)
table(pred)
table(Songs_test$Class, pred)
confusionMatrix(data = pred, reference = Songs_test$Class, mode = "prec_recall")
plot(caret_knn)




