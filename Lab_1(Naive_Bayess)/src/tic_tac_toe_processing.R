library(e1071)
library(ggplot2)
library(pROC)
library(yardstick)
library(dplyr)
library(correlationfunnel)
library(comprehenr)

# tic_tac_toe
set.seed(112)
A_raw <- read.table("Tic_tac_toe.txt", sep = ",", stringsAsFactors = TRUE)

A_size <- dim(A_raw)[1]
A_size
error_type_1 <- c()
error_type_2 <- c()
test_factors <- seq(0.1, 0.9, by=0.1)

TP <- c()
FN <- c()
FP <- c()
TN <- c()

true_classes <- c()
predictions <- c()
predictions_notraw <- c()
i <- 1
test_factors <- seq(0.1, 0.9, by=0.1)
for (factor_range in test_factors){
  idx <- sample(A_size, factor_range * A_size)
  tic_tac_train <- A_raw[-idx, ]
  tic_tac_test <- A_raw[idx, ]
  
  model <- naiveBayes(V10 ~ ., data = tic_tac_train)
  prediction <- predict(model, tic_tac_test)
  prediction_raw <- predict(model, tic_tac_test, type = 'raw')
  prediction_table <- table(prediction, tic_tac_test$V10)
  
  TP <- append(TP, prediction_table[1, 1])
  FP <- append(FP, prediction_table[1, 2])
  FN <- append(FN, prediction_table[2, 1])
  TN <- append(TN, prediction_table[2, 2])
  
  true_classes[[as.character(i)]] <- tic_tac_test$V10
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
positives <- c()
negatives <- c()
for (i in idxs){
  tmp_test <- true_classes[[as.character(i)]]
  tmp_prediction <- predictions[[as.character(i)]]
  binarized_tmp_test <- binarize(data.frame(tmp_test))
  first <- as.numeric(unlist(binarized_tmp_test[1]))
  second <- as.numeric(unlist(binarized_tmp_test[2]))
  roc_positive <- roc(first, tmp_prediction[, 1])
  roc_negative <- roc(second, tmp_prediction[, 2])
  positives[[as.character(i)]] <- roc_positive
  negatives[[as.character(i)]] <- roc_negative
}

roc_labels <- to_list(for(fact in seq(0.1, 0.9, by = 0.1)) as.character(fact))
i <- 1
positives_lst <- c()
negatives_lst <- c()
for (lab in roc_labels){
  auc_roc_positive <- round(auc(positives[[as.character(i)]]), digits = 5)
  positives_lst[[paste0(lab, " , auc = ", auc_roc_positive)]] <-
    positives[[as.character(i)]]
  auc_roc_negative <- round(auc(negatives[[as.character(i)]]), digits = 5)
  negatives_lst[[paste0(lab, " , auc = ", auc_roc_negative)]] <-
    negatives[[as.character(i)]]
  i <- i + 1
}

plt_positive <- ggroc(positives_lst, legacy.axes = TRUE) + 
  ggtitle("ROC-curves for classfication positive game results") +
  geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
              linetype = "dashed") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue")) 
ggsave("tic_tac_positive_roc.jpg")
plt_negative <- ggroc(negatives_lst, legacy.axes = TRUE) + 
  ggtitle("ROC-curves for classfication negative game results") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       linetype = "Test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue")) +
  geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
              linetype = "dashed")
ggsave("tic_tac_negative_roc.jpg")

#Recall-Precision
pr_df <- data.frame()
for (i in seq(1, 9, by = 1)){
  tmp_df <- data.frame(true = true_classes[[as.character(i)]],
                       positive = predictions[[as.character(i)]][, 2],
                       negative = predictions[[as.character(i)]][, 1],
                       predicted = predictions_notraw[[as.character(i)]],
                       resample = rep(as.character(0.1 * i),
                                      times = length(true_classes[[as.character(i)]])))
  pr_df <- rbind(pr_df, tmp_df)
}

pr_df %>%
  group_by(resample) %>%
  pr_curve(true, positive) %>%
  autoplot() + labs(title = "PR-curve for positive results")
ggsave("tic_tac_positive_pr.jpg")

pr_df %>%
  group_by(resample) %>%
  pr_curve(true, negative) %>%
  autoplot() + labs(title = "PR-curve for negative results")
ggsave("tic_tac_negative_pr.jpg")

for (size in seq(0.1, 0.9, by = 0.1)){
  print(paste0("При доле выборки в ", as.character(size)))
  size_df <- filter(pr_df, resample == as.character(size))
  auc_positive <- pr_auc(size_df, true, positive)
  auc_negative <- pr_auc(size_df, true, negative)
  print(paste0("Для положительного результата: auc == ",
               auc_positive$.estimate))
  print(paste0("Для отрицательного результата: auc == ",
               auc_negative$.estimate))
}
