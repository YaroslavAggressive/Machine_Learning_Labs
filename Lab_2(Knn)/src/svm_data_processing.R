library(kknn)
library(ggplot2)
library(hash)

test_name <- "svmdata4test.txt"
train_name <- "svmdata4.txt"
test_svm <- read.table(test_name)
train_svm <- read.table(train_name)
train_svm$Colors <- as.factor(train_svm$Colors)
test_svm$Colors <- as.factor(test_svm$Colors)

# initial dataset visualization
names <- c("train", "test", "total")
all_datasets <- list(train_svm, test_svm, rbind(train_svm, test_svm))
i <- 1
for (dataset in all_datasets){
  plt_train <- ggplot(data = dataset, aes(x = X1, y = X2, color = Colors)) +
    geom_point(aes(x = X1, y = X2, color = Colors, shape = Colors), size = 4.0) +
    geom_point(aes(x = X1, y = X2, shape = factor(Colors)),size = 1.5,
               color  = "grey90") +
    labs(title = paste0(names[i],
                        " sample of points dataset"))
  ggsave(paste0("true_", names[i], "_points.jpg"))
  i <- i + 1
}

best_kernel_i <- 0
kernels <- c("triangular", "rectangular", "epanechnikov", "gaussian")
for (kernel in kernels){
  accuracy <- c()
  k_values <- seq(2, 25, by = 1)
  for (k in k_values){
    svm_fit <- kknn(Colors ~., train_svm, test_svm, distance = 2, k = k,
                    kernel = kernel)
    fit <- fitted(svm_fit)
    tbl <- table(test_svm$Colors, svm_fit$fitted.values)
    accuracy <- append(accuracy, sum(diag(tbl)) / sum(tbl))
  }
  optimal_k <- k_values[match(max(accuracy), accuracy)]
  
  kernel_plot <- ggplot(data.frame(k = k_values, accuracy = accuracy),
                        aes(x = k, y = accuracy)) +
    geom_line(color = "darkblue") +
    geom_point() +
    geom_smooth(se=FALSE) +
    labs(title = "Dependence on k-value on accuracy of classification")
  ggsave(paste0(kernel, "_k_accuracy.jpg"))
  
  print(paste0("For kernel ", kernel, " optimal k == ", 
               as.character(optimal_k), ". Optimal accuracy == ", 
               as.character(max(accuracy))))
}

