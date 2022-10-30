library(e1071)
library(ggplot2)

set.seed(111)
# tic_tac_toe

A_raw <- read.table("Tic_tac_toe.txt", sep = ",", stringsAsFactors = TRUE)

A_size <- dim(A_raw)[1]
A_size
error_type_1 <- c()
error_type_2 <- c()
test_factors <- seq(0.1, 0.9, by=0.1)

for (factor_range in test_factors){
  idx <- sample(1: A_size, factor_range * A_size)
  A_train <- A_raw[-idx, ]
  A_test <- A_raw[idx, ]
  
  model <- naiveBayes(V10 ~., A_train)
  
  prediction <- predict(model, A_test[1: (length(A_test) - 1)])
  prediction_table <- table(prediction, A_test$V10)
  
  error_type_1 <- append(error_type_1, prediction_table[2, 1])
  error_type_2 <- append(error_type_2, prediction_table[1, 2])
}

plt_1 <- ggplot(data = data.frame(test_factors, error_type_1),
                aes(x = test_factors, y = error_type_1)) +
  geom_line() +
  geom_point() + 
  theme_light() + 
  labs(x = "Part of data in test sample", y = "Type 1 Error",
       title = "Dependence of type 1 error on test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"))
ggsave("tic_tac_error_1.jpg")
plt_2 <- ggplot(data = data.frame(test_factors, error_type_2),
                aes(x = test_factors, y = error_type_2)) +
  geom_line() +
  geom_point() + 
  theme_light() + 
  labs(x = "X2", y = "Frequency",
       title = "Dependence of type 2 error on test sample size") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"))
ggsave("tic_tac_error_2.jpg")
