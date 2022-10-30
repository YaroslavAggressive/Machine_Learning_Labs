library(e1071)
library(kernlab)
library(ggplot2)

set.seed(111)
# spam
data(spam)

error_type_1 <- c()
error_type_2 <- c()
test_factors <- seq(0.1, 0.9, by=0.1)
for (factor_range in test_factors){
  idx <- sample(1: dim(spam)[1], factor_range * dim(spam)[1])
  spam_train <- spam[-idx, ]
  spam_test <- spam[idx, ]
  
  model <- naiveBayes(type ~ ., data = spam_train)
  prediction <- predict(model, spam_test)
  prediction_table <- table(prediction, spam_test$type)
  
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
ggsave("spam_error_1.jpg")

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
ggsave("spam_error_2.jpg")
