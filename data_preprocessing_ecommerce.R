# Data Preprocessing Template

# Import the dataset
dataset = read.csv('new_e-commerce-data.csv')
d = read.csv('ecommerce_sample.csv')
ecommerce_sample <- read.csv('ecommerce_sample.csv')
homedepot_sample = read.csv("homedepot_samplev1.csv")
homedepot_sample$product_title1 <- sort(homedepot_sample$product_title1)

# Missing Data
#dataset$Age = ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)
#dataset$Salary = ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function (x) mean(x, na.rm = TRUE)), dataset$Salary)
#dataset$Salary <- ifelse(is.na(dataset$Salary), mean(dataset$Salary, na.rm = TRUE), dataset$Salary)

  

# Categorical Data
#dataset$Country <- factor(dataset$Country, levels = c('France', 'Spain', 'Germany'), labels = c(1,2,3))
#dataset$Purchased <- factor(dataset$Purchased, levels = c('No', 'Yes'), labels = c(0,1))

# Splitting the dataset into training set and test set
set.seed(123)
split <- sample.split(dataset$product_title, SplitRatio = 0.25595347789)# Returns a logical vector where 80% of values are TRUE.
split <- logical(length = 25000)
split <- !split #changes all entries in logical vector split to true
split <- ifelse(isTRUE(split), TRUE, TRUE)

# dataset$Purchased is dependent variable
training_set <- subset(dataset, split == TRUE)# Creates a subset of dataset of the observations corresponding to the TRUE values in split
test_set <- subset(dataset, split == FALSE)# Creates a subset of dataset of the observations corresponding to the FALSE values in split
ecommerce_sample <- subset(dataset, split == FALSE)
homedepot_sample1 <- subset(dataset, split == TRUE)


# Feature Scaling # datafram[row, column]
training_set[, 2:3] <- scale(training_set[, 2:3]) #Scale can only function with numeric vectors
test_set[, 2:3] <- scale(test_set[, 2:3])
