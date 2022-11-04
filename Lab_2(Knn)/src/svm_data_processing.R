library(kknn)
library(ggplot2)
library(hash)

test_name <- "svmdata4test.txt"
train_name <- "svmdata4.txt"
test_svm <- read.table(test_name)
train_svm <- read.table(train_name)
train_svm$Colors <- as.factor(train_svm$Colors)
test_svm$Colors <- as.factor(test_svm$Colors)

best_kernel_i <- 0
kernels <- c("triangular", "rectangular", "epanechnikov", "optimal")
for (kernel in kernels){
  accuracy <- c()
  iters <- 5
  k_values <- seq(2, 25, by = 1)
  for (k in k_values){
    svm_fit <- kknn(Colors ~., train_svm, test_svm, distance = 2, k = k,
                    kernel = kernel)
    fit <- fitted(svm_fit)
    tbl <- table(test_svm$Colors, fit)
    accuracy <- append(accuracy, (sum(diag(tbl))) /sum(tbl))
  }
  optimal_k <- k_values[match(min(accuracy), accuracy)]
  print(paste0("For kernel ", kernel, " optimal k == ", 
               as.character(optimal_k), ". Optimal accuracy == ", 
               as.character(min(accuracy))))
}

# initial dataset visualization
plt_train <- ggplot(data = train_svm, aes(x = X1, y = X2, color = Colors)) +
  geom_point() +
  labs(title = "Train sample of points dataset")
ggsave("true_train_points.jpg")
plt_test <- ggplot(data = test_svm, aes(x = X1, y = X2, color = Colors)) +
  geom_point() +
  labs(title = "Test sample of points dataset")
ggsave("true_test_points.jpg")
plt_total <- ggplot(data = rbind(train_svm, test_svm), 
                    aes(x = X1, y = X2, color = Colors)) +
  geom_point() +
  labs(title = "Total sample of points dataset")
ggsave("true_total_points.jpg")
