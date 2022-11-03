library(e1071)
library(ggplot2)
library(tibble)
library(cvms)
library(ggimage)
library(broom)
library(ggpubr)
library(rsvg)

# titanic
test_name <- "titanic/test.csv"
train_name <- "titanic/train.csv"
titanic_test <- read.csv(test_name, sep = ",")
titanic_train <- read.csv(train_name, sep = ",")
model <- naiveBayes(Survived ~., data = titanic_train)
prediction_titanic <- predict(model, titanic_test)
prop.table(table(titanic_train$Survived))
prop.table(table(prediction_titanic))

# normal distributions

n <- 50  # number of points in each part

# numeric characteristic of random value for first 50 points
a1 <- 10
a2 <- 14
sigma12 <- 4
first_part <- data.frame(rnorm(n, mean = a1, sd = sigma12),
                         rnorm(n, mean = a2, sd = sigma12),
                         rep(c("N(10, 16)"), each = n),
                         rep(c("N(14, 16)"), each = n),
                         rep(c(as.factor("Class 1")), each = 50))

# numeric characteristic of random value for second 50 points
a1 <- 20
a2 <- 18
sigma12 <- 3
second_part <- data.frame(rnorm(n, mean = a1, sd = sigma12),
                          rnorm(n, mean = a2, sd = sigma12),
                          rep(c("N(20, 9)"), each = n),
                          rep(c("N(18, 9)"), each = n),
                          rep(c(as.factor("Class 2")), each = n))

colnames(second_part) <- c("X1", "X2", "X1_Distribution", "X2_Distribution", "Type")
colnames(first_part) <- c("X1", "X2", "X1_Distribution", "X2_Distribution", "Type")

total_data <- rbind(first_part, second_part)

# иллюстрация данных

plt_1 <- ggplot() + 
  geom_histogram(data = total_data, aes(x = X1, y = ..count../sum(..count..)),
                 bins = 20, color="black", fill="blue") +
  geom_density(data = first_part, aes(x = X1, color = X1_Distribution), 
               fill = "green", lwd = 0.8, linetype = 2, alpha = 0.1,
               show.legend = TRUE) + 
  geom_density(data = second_part, aes(x = X1, color = X1_Distribution), 
               fill = "red", lwd = 0.8, linetype = 2, alpha = 0.1,
               show.legend = TRUE) +
  theme_light() + 
  labs(x = "X2", y = "Frequency",
       title = "Frequency histogram of the sample by feature X1") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"))
ggsave("histogram_X1.jpg")

plt_2 <- ggplot() + 
  geom_histogram(data = total_data, aes(x = X2, y = ..count../sum(..count..)),
                 bins = 20, color="black", fill="blue") + 
  geom_density(data = first_part, aes(x = X2, color = X2_Distribution),
               fill = "green", lwd = 0.8, linetype = 2, alpha = 0.1,
               show.legend = TRUE) + 
  geom_density(data = second_part, aes(x = X2, color = X2_Distribution),
               fill = "red", lwd = 0.8, linetype = 2, alpha = 0.1,
               show.legend = TRUE) +
  theme_light() + 
  labs(x = "X2", y = "Frequency", 
       title = "Frequency histogram of the sample by feature X2") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"))
ggsave("histogram_X2.jpg")

# basic classifier, 25 test and 75 train individuals

test_size <- n / 2
idx_test <- sample(1: (2 * n), test_size)
test_df <- total_data[idx_test, ]
train_df <- total_data[-idx_test, ]
drop_columns <- c("X1_Distribution", "X2_Distribution")

# положение точек в пространстве
plt_points <- ggplot(total_data[, !names(train_df) %in% drop_columns]) +
  geom_point(aes(x = X1, y = X2, color = Type, 
                 shape = Type), size = 4.0) + 
  geom_point(aes(x = X1, y = X2, shape = factor(Type)),
             size = 1.5, color  = "grey90") +
  theme_light() + 
  labs(x = "X1", y = "X1", title = "Distribution of the dataset by class") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"))
ggsave("distribution_X1_X2.jpg")

model <- naiveBayes(Type ~., train_df[, !names(train_df) %in% drop_columns])
drop_columns <- append(drop_columns, "Type")
prediction <- predict(model, test_df[, !names(train_df) %in% drop_columns])
result_table <- table(prediction, test_df$Type)

# матрица ошибок, используя tibble
result_tibble <- tibble("True" = as.integer(test_df[, 5]),
                        "Prediction" = as.integer(prediction))
cfm <- confusion_matrix(targets = result_tibble$True, 
                        predictions = result_tibble$Prediction)
plot_confusion_matrix(
  cfm$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)
ggsave("cfm_normal.jpg")

# итоги классификации

plt_true_classes <- ggplot(test_df) +
  geom_point(aes(x = X1, y = X2, color = Type, shape = Type), size = 4.0) +
  geom_point(aes(x = X1, y = X2, shape = Type), size = 1.5, color = "grey90") +
  theme_light() + labs(x = "X1", y = "X1", title = "True Classification") +
  theme(legend.background = element_rect(fill="lightblue",
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"),
        legend.position="bottom")
ggsave("true_classification.jpg")

plt_prediction_classes <- ggplot(cbind(test_df[1: (length(test_df) - 3)],
                                      Type = prediction)) +
  geom_point(aes(x = X1, y = X2, color = Type, shape = Type), size = 4.0) +
  geom_point(aes(x = X1, y = X2, shape = Type), size = 1.5, color = "grey90") +
  theme_light() + labs(x = "X1", y = "X1", title = "Prediction") +
  theme(legend.background = element_rect(fill="lightblue", 
                                         size=0.5, linetype="solid",
                                         colour ="darkblue"),
        legend.position="bottom")

ggsave("prediction.jpg")

plt_classification <- ggarrange(plt_true_classes, plt_prediction_classes, 
                                ncol = 2, nrow = 1)
ggsave("comparison_test_sample.jpg")
                                    