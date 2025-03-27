#loading libraries
library(class)
library(caret)
library(e1071)
library(dplyr)
library(readr)
library(fastDummies)
library(MLmetrics)
library(generics)
library(caret)
library(dplyr)
library(neuralnet)
library(countrycode)
library(ggthemes)

# importing  data
salaries <- read.csv('Salaries.fixed.csv')
summary(salaries)

# converting categorical data into dummy variables
# not selecting employee residence
# will need to change remote ratio to categorical as it is numeric
salaries <- salaries %>%
  mutate(remote_ratio = as.character(remote_ratio))
dummys <- dummy_cols(salaries[c(2:3,5:8)])

# scaling salaries
salaries <- salaries %>%
  mutate(scaled.salaries = scale(salary_in_usd))

# binding dummys to salaries
salaries.fixed <- cbind(salaries = salaries$scaled.salaries,
                        dummys[-c(1:6)])

# removing spaces in the column names and replacing with "-"
colnames(salaries.fixed) <- gsub(" ", "_", colnames(salaries.fixed))

# partitioning data
set.seed(1)
train.index <- sample(row.names(salaries.fixed), 0.6*dim(salaries.fixed)[1])
valid.index <- setdiff(row.names(salaries.fixed), train.index)  
train.df <- salaries.fixed[train.index, ]
valid.df <- salaries.fixed[valid.index, ]

# saving mean and standard deviation of unscaled salaries
mean.salary <-mean(salaries$salary_in_usd)
sd.salary <- sd(salaries$salary_in_usd)


# trying different values to optimise for k
accuracy.df <- data.frame()
set.seed(123)
for (i in 1:15) {
  knn.pred <- class::knn(train = train.df[, -1],
                         test = valid.df[, -1],
                         cl = train.df$salaries, k = i)
  knn.pred.salary <- (as.numeric(as.character(knn.pred))*sd.salary)+mean.salary
  valid.salary <- valid.df$salaries*sd.salary+mean.salary
  accuracy.df[i,1] <- MAPE(knn.pred.salary, valid.salary)
  accuracy.df[i,2] <- RMSE(knn.pred.salary, valid.salary)
  accuracy.df[i,3] <- i
}
colnames(accuracy.df) <- c("MAPE", "RMSE", "k")

# plotting accuracy results to show best k value
theme_set(theme_solarized())
ggplot(accuracy.df, aes(x = k, y = MAPE)) +
  geom_line(size = .4) +
  geom_point(color = "red", size = 3) +
  labs(title = "MAPE for each k")

ggplot(accuracy.df, aes(x = k, y = RMSE)) +
  geom_line(size = .4) +
  geom_point(color = "red", size = 3) +
  labs(title = "RMSE for each k")

# k = 3 has the lowest errors so will train the model on that
knn.pred <- knn(train = train.df[,-1],
                test = valid.df[,-1],
                cl = train.df$salaries, k = 3)

# knn model's prediction
knn.pred.salary <- (as.numeric(as.character(knn.pred))*sd.salary)+mean.salary
valid.salary <- (valid.df$salaries*sd.salary)+mean.salary

# some error metrics here
RMSE(knn.pred.salary, valid.salary)
MAPE(knn.pred.salary, valid.salary)

# plotting difference in predicted salaries vs actual salaries from validation set
knn.salary.diff <- data.frame(number = 1:478,
                              difference = (knn.pred.salary-valid.salary)/valid.salary)

ggplot(knn.salary.diff, aes(x = number, y = difference)) +
  geom_point(color = "blue", size = 2, alpha = 0.7) +  
  geom_hline(yintercept = 0.2, color = "red", linetype = "dashed", linewidth = 1) +  
  geom_hline(yintercept = -0.2, color = "red", linetype = "dashed", linewidth = 1) + 
  geom_hline(yintercept = 0, color = "green", linewidth = 1) + 
  labs(
    title = "Percentage difference in Predicted vs Actual Salary",
    x = "Validation Set",
    y = "Percentage error"
  ) +
  ylim(1,-2)


# moving on to clustering
d.norm <- dist(salaries.fixed, method = "euclidean")
cluster1 <- hclust(d.norm,method='ward.D')
plot(cluster1, hang= -1, ann= TRUE)

clust_types<-cutree(cluster1,k=2)

# k means clustering
km <- kmeans(salaries.fixed,2)
table(clust_types,km$cluster)

# plot to describe cluster centroids characteristics
# have to look at the observations that have a big difference
plot(c(0), xaxt = 'n', ylab = "", type = "l",
     ylim = c(min(km$centers), max(km$centers)), xlim = c(0, 27))


# label x-axes
axis(1, at = c(1:32), labels = names(salaries.fixed))
for (i in c(1:2))
  lines(km$centers[i,], lty = i, lwd = 2, col = ifelse(i %in% c(1, 3, 5),
                                                       "black",
                                                       "dark grey"))

# training a neural net
model.nn <- neuralnet(salaries ~ .,
                      data = train.df, linear.output = T,hidden = 3)

plot(model.nn)

# applying the neural net model to the validation data set and checking errors
valid.pred.nn <- compute(model.nn, valid.df)
RMSE(valid.pred.nn$net.result, valid.df$salaries)

# linear model
model.lm <- lm(salaries ~ .,
               data = train.df)

# applying linear model to validation set and checking error
salary.lm.pred <- predict.lm(model.lm, valid.df[,-c(1)])
MAPE(salary.lm.pred, valid.salary)
RMSE(salary.lm.pred, valid.salary)
