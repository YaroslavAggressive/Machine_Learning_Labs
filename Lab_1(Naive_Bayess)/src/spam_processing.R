library(e1071)
library(kernlab)
library(ggplot2)
library(pROC)
library(yardstick)
library(dplyr)
library(correlationfunnel)
library(comprehenr)

set.seed(112)
data(spam)

TP <- c()
FN <- c()
FP <- c()
TN <- c()

n <- dim(spam)[1]
true_classes <- c()
predictions <- c()
predictions_notraw <- c()
i <- 1
test_factors <- seq(0.1, 0.9, by=0.1)
for (factor_range in test_factors){
  idx <- sample(n, factor_range * n)
  spam_train <- spam[-idx, ]
  spam_test <- spam[idx, ]
  
  model <- naiveBayes(type ~ ., data = spam_train)
  prediction <- predict(model, spam_test)
  prediction_raw <- predict(model, spam_test, type = 'raw')
  prediction_table <- table(prediction, spam_test$type)
  
  TP <- append(TP, prediction_table[1, 1])
  FP <- append(FP, prediction_table[1, 2])
  FN <- append(FN, prediction_table[2, 1])
  TN <- append(TN, prediction_table[2, 2])
  
  true_classes[[as.character(i)]] <- spam_test$type
  predictions[[as.character(i)]] <- prediction_raw
  predictions_notraw[[as.character(i)]] <- prediction
  i <- i + 1
}

# for Recall-Precision Curve
recalls <- TP / (TP + FP)
precisions <- TP / (TP + FN)
# for ROC-Curve
TPR <- recalls
FPR <- FP / (FP + TN)

# ROC
idxs <- seq(1, 9, by = 1)
spams <- c()
nonspams <- c()
for (i in idxs){
  tmp_test <- true_classes[[as.character(i)]]
  tmp_prediction <- predictions[[as.character(i)]]
  binarized_tmp_test <- binarize(data.frame(tmp_test))
  first <- as.numeric(unlist(binarized_tmp_test[1]))
  second <- as.numeric(unlist(binarized_tmp_test[2]))
  roc_nonspam <- roc(first, tmp_prediction[, 1])
  roc_spam <- roc(second, tmp_prediction[, 2])
  spams[[as.character(i)]] <- roc_spam
  nonspams[[as.character(i)]] <- roc_nonspam
}

roc_labels <- to_list(for(fact in seq(0.1, 0.9, by = 0.1)) as.character(fact))
i <- 1
spams_lst <- c()
nonspams_lst <- c()
for (lab in roc_labels){
  auc_roc_spams <- round(auc(spams[[as.character(i)]]), digits = 5)
  spams_lst[[paste0(lab, " , auc = ", auc_roc_spams)]] <-
    spams[[as.character(i)]]
  auc_roc_nonspams <- round(auc(nonspams[[as.character(i)]]), digits = 5)
  nonspams_lst[[paste0(lab, " , auc = ", auc_roc_nonspams)]] <-
    nonspams[[as.character(i)]]
  i <- i + 1
}


plt_spam <- ggroc(spams_lst, legacy.axes = TRUE) + 
  ggtitle("ROC-curves for classfication 'spam' messages") +
  geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
              linetype = "dashed") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue")) 
ggsave("spam_roc.jpg")
plt_nonspam <- ggroc(nonspams_lst, legacy.axes = TRUE) + 
  ggtitle("ROC-curves for classfication 'nonspam' messages") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue")) +
  geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
              linetype = "dashed")
ggsave("nonspam_roc.jpg")

#Recall-Precision
pr_df <- data.frame()
for (i in seq(1, 9, by = 1)){
  tmp_df <- data.frame(true = true_classes[[as.character(i)]],
                       spam = predictions[[as.character(i)]][, 2],
                       nonspam = predictions[[as.character(i)]][, 1],
                       predicted = predictions_notraw[[as.character(i)]],
                       resample = rep(as.character(0.1 * i),
                                      times = length(true_classes[[as.character(i)]])))
  pr_df <- rbind(pr_df, tmp_df)
}

pr_df %>%
  group_by(resample) %>%
  pr_curve(true, spam) %>%
  autoplot() + labs(title = "PR-curve for spam messages")
ggsave("spam_pr.jpg")

pr_df %>%
  group_by(resample) %>%
  pr_curve(true, nonspam) %>%
  autoplot() + labs(title = "PR-curve for nonspam messages")
ggsave("nonspam_pr.jpg")

for (size in seq(0.1, 0.9, by = 0.1)){
  print(paste0("При доле выборки в ", as.character(size)))
  size_df <- filter(pr_df, resample == as.character(size))
  auc_spam <- pr_auc(size_df, true, spam_)
  auc_nonspam <- pr_auc(size_df, true, nonspam_)
  print(paste0("Для спама: auc == ", auc_spam$.estimate))
  print(paste0("Для НЕ спама: auc == ", auc_nonspam$.estimate))
}
