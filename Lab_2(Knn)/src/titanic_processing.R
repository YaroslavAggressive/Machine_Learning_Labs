library(kknn)
library(ggplot2)
library(knitr)
library(class)

test_name <- "test.csv"
train_name <- "train.csv"
valid_df <- read.csv(test_name, sep = ",")
learn_df <- read.csv(train_name, sep = ",")
learn_df$Survived <- as.factor(learn_df$Survived)

kernels <- c("rectangular", "triangular", "epanechnikov", "gaussian")
k_for_kernel <- c()
for (kernel in kernels){
  model <- train.kknn(Survived ~., learn_df, kmax = 25, kernel = kernel,
                      distance = 2)
  k_for_kernel <- append(k_for_kernel, titanic_train$best.parameters$k)
}

for (kernel in kernels) {
  model <- train.kknn(Survived ~ Age + Pclass + Fare + Sex + SibSp,
                      learn_df, kernel = kernel, distance = 2)
  prediction <- predict(model, valid_df)
  print(paste0("Для ядра ", kernel, ": "))
  print(prop.table(table(learn_df$Survived)))
  print(prop.table(table(prediction)))
}

