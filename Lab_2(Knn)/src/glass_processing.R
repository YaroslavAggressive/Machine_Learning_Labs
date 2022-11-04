library(kknn)
library(ggplot2)

data(glass)
colnames(glass)
glass$Id <- NULL

kernels <- c("rectangular", "triangular", "epanechnikov", "biweight",
             "triweight", "cos", "inv", "gaussian", "optimal")
k_values <- seq(2, 25, by = 1)
iters <- 10
glass_n <- dim(glass)[1]
col_n <- dim(glass)[2]
valid_size <- ceiling(0.2 * glass_n)

result <- c()

for (kernel in kernels){
  kernel_k_accuracy <- c()
  for (k in k_values){
    values_for_mean <- c()
    for (iter in iters){
      test_idx <- sample(1: glass_n, valid_size)
      learn_sample <- glass[-test_idx, ]
      valid_sample <- glass[test_idx, ]
      knn_fit <- kknn(Type ~., learn_sample, valid_sample, distance = 2, k = k,
                      kernel = kernel)
      fit <- fitted(knn_fit)
      tbl <- table(valid_sample$Type, fit)
      values_for_mean <- append(values_for_mean, 
                                (sum(diag(tbl))) /sum(tbl))
    }
    kernel_k_accuracy <- append(kernel_k_accuracy, mean(values_for_mean))
  }
  result[[kernel]] <- kernel_k_accuracy
}

for (kernel in kernels){
  
}
result
glass

# dependence of distance-parameter on classification accuracy

distances <- seq(1, 25, by = 1)

# finding dependence of each factor on classification error

iters <- 5
factor_names <- colnames(glass[, -col_n])
accuracies_for_hist <- c()
for (colname in factor_names){
  temp_df <- glass[!names(df) %in% c(colname)]
  values_for_mean <- c()
  for (iter in iters){
    test_idx <- sample(1: glass_n, valid_size)
    learn_sample <- glass[-test_idx, ]
    valid_sample <- glass[test_idx, ]
    knn_fit <- kknn(Type ~., learn_sample, valid_sample, distance = 2, k = k,
                    kernel = kernel)
    fit <- fitted(knn_fit)
    tbl <- table(valid_sample$Type, fit)
    values_for_mean <- append(values_for_mean, 
                              (sum(diag(tbl))) /sum(tbl))
  }
  accuracies_for_hist <- append(accuracies_for_hist, round(mean(values_for_mean), 4))
}

# plot histogram of accuracies

bar_data <- data.frame(name = factor_names,
                       value = accuracies_for_hist)
accuracies_hist <- ggplot(bar_data, aes(x = name, y = value)) +
  geom_bar(stat = "identity", , width = 0.5, fill = "darkblue") + 
  labs(x = "Factor name", 
       y = "Classification accuracy without factor",
       title = "Dependence of each factor on classification accuracy") +
  geom_text(aes(label = value), vjust = -0.2, size = 5,
            position = position_dodge(0.9)) +
  ylim(0, max(bar_data$value)*1.1)
ggsave("factors_bar.jpg")

# test classification of one glass element
test_example <- data.frame(RI = 1.516, Na = 11.7, Mg = 1.01, Al = 1.19, 
                           Si = 72.59, K = 0.43, Ca = 11.44, Ba = 0.02, Fe = 0.1)
