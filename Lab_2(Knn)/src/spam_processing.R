library(kknn)
library(kernlab)
library(ggplot2)
library(reshape)
library(pROC)
library(yardstick)
library(dplyr)
library(correlationfunnel)
library(comprehenr)

# spam
set.seed(111)
data(spam)

# проверим все ядра
n <- dim(spam)[1]
kernels = c("triangular", "rectangular", "epanechnikov", "optimal")
factor_sizes <- seq(0.1, 0.9, by = 0.05)

# finding best k and min accuracies
kernel_accuracy <- c()
best_k <- c()
for (kernel in kernels){
    fit_train <- train.kknn(type ~., spam, kmax = 25, distance = 2,
                            kernel = kernel)
    kernel_accuracy[kernel] <- min(fit_train$MISCLASS)
    best_k[kernel] <- fit_train$best.parameters$k
}

# for accuracy plots for each test sample size
accuracies <- c()
iters <- 10
for (kernel in kernels){
  kernel_accuracy <- c()
  for (test_factor in factor_sizes){
    accuracy_for_mean <- c()
    for (iter in iters){  # усреднее по 5 итерациям классификации для уменьшения случайности оценки точности
      test_idxs <- sample(1: n, ceiling(n * test_factor))
      learn_sample <- spam[test_idxs, ]
      valid_sample <- spam[-test_idxs, ]
      
      fit_train <- kknn(type ~., learn_sample, valid_sample, 
                        k = best_k[[kernel]], distance = 2, kernel = kernel)
      fit <- fitted(fit_train)
      tbl <- table(valid_sample$type, fit)
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
      labs(x = "Part of all messages in training sample", 
           y = "Minimal misclassification value",
           title = "Dependence of classification error on test sample size") +
      theme(legend.background = element_rect(fill="cyan",
                                             size=0.5, linetype="longdash",
                                             colour ="darkblue"))
    ggsave(paste0(kernel, "_accuracy.jpg"))
}

# for creating ROC and PR curves
test_factors <- seq(0.1, 0.9, by=0.1)
for (kernel in kernels){
  true_classes <- c()
  predictions <- c()
  predictions_raw <- c()
  i <- 1
  for (factor_range in test_factors){
    idx <- sample(n, factor_range * n)
    spam_train <- spam[-idx, ]
    spam_test <- spam[idx, ]
    
    kknn_results <- kknn(type ~ ., spam_train, spam_test, 
                  k = best_k[[kernel]], kernel = kernel, distance = 2)
    prediction <- kknn_results$fitted.values
    prediction_raw <- kknn_results$prob
    prediction_table <- table(prediction, spam_test$type)
    
    true_classes[[as.character(i)]] <- spam_test$type
    predictions_raw[[as.character(i)]] <- prediction_raw
    predictions[[as.character(i)]] <- prediction
    i <- i + 1
  }
  
  idxs <- seq(1, 9, by = 1)
  spams <- c()
  nonspams <- c()
  for (i in idxs){
    tmp_test <- true_classes[[as.character(i)]]
    tmp_prediction <- predictions_raw[[as.character(i)]]
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
    ggtitle(paste0("ROC-curves for classfication 'spam' messages for kernel ", kernel)) +
    geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
                linetype = "dashed") +
    labs(x = "1 - Specificity",
         y = "Sensitivity",
         linetype = "Test sample size") +
    theme(legend.background = element_rect(fill="lightblue",
                                           size=0.5, linetype="solid",
                                           colour ="darkblue")) 
  ggsave(paste0(kernel, "_spam_roc.jpg"))
  plt_nonspam <- ggroc(nonspams_lst, legacy.axes = TRUE) + 
    ggtitle(paste0("ROC-curves for classfication 'nonspam' messages for kernel ", kernel)) +
    labs(x = "1 - Specificity",
         y = "Sensitivity",
         linetype = "Test sample size") +
    theme(legend.background = element_rect(fill="lightblue",
                                           size=0.5, linetype="solid",
                                           colour ="darkblue")) +
    geom_abline(intercept = 1, slope = 1, color = 'grey', size = 0.5,
                linetype = "dashed")
  ggsave(paste0(kernel, "_nonspam_roc.jpg"))
  
  #Recall-Precision
  pr_df <- data.frame()
  for (i in seq(1, 9, by = 1)){
    tmp_df <- data.frame(true = true_classes[[as.character(i)]],
                         spam = predictions_raw[[as.character(i)]][, 2],
                         nonspam = predictions_raw[[as.character(i)]][, 1],
                         predicted = predictions[[as.character(i)]],
                         resample = rep(as.character(0.1 * i),
                                        times = length(true_classes[[as.character(i)]])))
    pr_df <- rbind(pr_df, tmp_df)
  }
  
  pr_df %>%
    group_by(resample) %>%
    pr_curve(true, spam) %>%
    autoplot() + labs(title = paste0("PR-curve for spam messages for kernel ", kernel))
  ggsave(paste0(kernel, "_spam_pr.jpg"))
  
  pr_df %>%
    group_by(resample) %>%
    pr_curve(true, nonspam) %>%
    autoplot() + labs(title = paste0("PR-curve for nonspam messages for kernel ", kernel))
  ggsave(paste0(kernel, "_nonspam_pr.jpg"))
  
  for (size in seq(0.1, 0.9, by = 0.1)){
    print(paste0("При доле выборки в ", as.character(size), ", для ядра ", kernel))
    size_df <- filter(pr_df, resample == as.character(size))
    auc_spam <- pr_auc(size_df, true, spam)
    auc_nonspam <- pr_auc(size_df, true, nonspam)
    print(paste0(kernel, ", Для спама: auc == ", auc_spam$.estimate))
    print(paste0(kernel, ", Для НЕ спама: auc == ", auc_nonspam$.estimate))
  }
  
}
  
