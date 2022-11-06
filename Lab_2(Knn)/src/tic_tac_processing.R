library(kknn)
library(ggplot2)
library(reshape)
library(pROC)
library(yardstick)
library(dplyr)
library(correlationfunnel)
library(comprehenr)

set.seed(111)
tic_tac_name <- "tic_tac_toe.txt"
A_raw <- read.table(tic_tac_name, sep = ",")
A_raw$V10 <- as.factor(tic_tac$V10)
A_size <- dim(tic_tac)[1]

kernels = c("triangular", "rectangular", "epanechnikov", "gaussian")
factor_sizes <- seq(0.1, 0.9, by = 0.05)

kernel_accuracy <- c()
best_k <- c()
for (kernel in kernels){
  fit_train <- train.kknn(V10 ~., tic_tac, kmax = 25, distance = 2,
                          kernel = kernel)
  kernel_accuracy[kernel] <- min(fit_train$MISCLASS)
  best_k[kernel] <- fit_train$best.parameters$k
}

accuracies <- c()
iters <- 10
for (kernel in kernels){
  kernel_accuracy <- c()
  for (test_factor in factor_sizes){
    accuracy_for_mean <- c()
    for (iter in iters){
      test_idxs <- sample(1: A_size, ceiling(A_size * test_factor))
      learn_sample <- A_raw[test_idxs, ]
      valid_sample <- A_raw[-test_idxs, ]
      
      fit_train <- kknn(V10 ~., learn_sample, valid_sample, 
                        k = best_k[[kernel]], distance = 2, kernel = kernel)
      fit <- fitted(fit_train)
      tbl <- table(valid_sample$V10, fit)
      accuracy_for_mean <- append(accuracy_for_mean, sum(diag(tbl))/ sum(tbl))
    }
    kernel_accuracy <- append(kernel_accuracy, mean(accuracy_for_mean))
  }
  kernel_results_acc[[kernel]] <- kernel_accuracy
}

for (kernel in kernels){
  accuracy <- kernel_results_acc[[kernel]]
  plt_accuracy <- ggplot(data = data.frame(factor_sizes, accuracy), 
                         aes(x = factor_sizes, y = accuracy)) +
    geom_line() +
    geom_point() + 
    theme_dark() +
    labs(x = "Part of games in training sample", 
         y = "Minimal misclassification value",
         title = "Dependence of classification error on test sample size") +
    theme(legend.background = element_rect(fill="cyan",
                                           size=0.5, linetype="longdash",
                                           colour ="darkblue"))
  ggsave(paste0(kernel, "_tic_tac_accuracy.jpg"))
}

test_factors <- seq(0.1, 0.9, by=0.1)
for (kernel in kernels){
  true_classes <- c()
  predictions <- c()
  predictions_raw <- c()
  i <- 1
  for (factor_range in test_factors){
    idx <- sample(A_size, factor_range * A_size)
    tic_tac_train <- A_raw[-idx, ]
    tic_tac_test <- A_raw[idx, ]
    
    kknn_results <- kknn(V10 ~ ., tic_tac_train, tic_tac_test, 
                         k = best_k[[kernel]], kernel = kernel, distance = 2)
    prediction <- kknn_results$fitted.values
    prediction_raw <- kknn_results$prob
    prediction_table <- table(prediction, tic_tac_test$V10)
    
    true_classes[[as.character(i)]] <- tic_tac_test$V10
    predictions[[as.character(i)]] <- prediction
    predictions_raw[[as.character(i)]] <- prediction_raw
    i <- i + 1
  }
  
  # ROC
  idxs <- seq(1, 9, by = 1)
  positives <- c()
  negatives <- c()
  for (i in idxs){
    tmp_test <- true_classes[[as.character(i)]]
    tmp_prediction <- predictions_raw[[as.character(i)]]
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
    ggtitle(paste0("ROC-curves for classfication positive game results for kernel ", kernel)) +
    geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
                linetype = "dashed") +
    labs(x = "1 - Specificity",
         y = "Sensitivity",
         linetype = "Test sample size") +
    theme(legend.background = element_rect(fill="lightblue",
                                           size=0.5, linetype="solid",
                                           colour ="darkblue")) 
  ggsave(paste0(kernel, "_tic_tac_positive_roc.jpg"))
  plt_negative <- ggroc(negatives_lst, legacy.axes = TRUE) + 
    ggtitle(paste0("ROC-curves for classfication negative game results for kernel ", kernel)) +
    labs(x = "1 - Specificity",
         y = "Sensitivity",
         linetype = "Test sample size") +
    theme(legend.background = element_rect(fill="lightblue",
                                           size=0.5, linetype="solid",
                                           colour ="darkblue")) +
    geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
                linetype = "dashed")
  ggsave(paste0(kernel, "_tic_tac_negative_roc.jpg"))
  
  #Recall-Precision
  pr_df <- data.frame()
  for (i in seq(1, 9, by = 1)){
    tmp_df <- data.frame(true = true_classes[[as.character(i)]],
                         positive = predictions_raw[[as.character(i)]][, 2],
                         negative = predictions_raw[[as.character(i)]][, 1],
                         predicted = predictions[[as.character(i)]],
                         resample = rep(as.character(0.1 * i),
                                        times = length(true_classes[[as.character(i)]])))
    pr_df <- rbind(pr_df, tmp_df)
  }
  
  pr_df %>%
    group_by(resample) %>%
    pr_curve(true, positive) %>%
    autoplot() + labs(title = paste0("PR-curve for positives results for kernel ", kernel))
  ggsave(paste0(kernel, "_tic_tac_positive_pr.jpg"))
  
  pr_df %>%
    group_by(resample) %>%
    pr_curve(true, negative) %>%
    autoplot() + labs(title = paste0("PR-curve for negatives results for kernel ", kernel))
  ggsave(paste0(kernel, "_tic_tac_negative_pr.jpg"))
  
  for (size in seq(0.1, 0.9, by = 0.1)){
    print(paste0("При доле выборки в ", as.character(size), " при ядре ", kernel))
    size_df <- filter(pr_df, resample == as.character(size))
    auc_positive <- pr_auc(size_df, true, positive)
    auc_negative <- pr_auc(size_df, true, negative)
    print(paste0(kernel, ", Для положительного результата: auc == ",
                 auc_positive$.estimate))
    print(paste0(kernel, ", Для отрицательного результата: auc == ",
                 auc_negative$.estimate))
  }
}

