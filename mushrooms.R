###  The mushroom classification problem is to determine whether a mushroom is edible or poisonous based on its observable features . 

# Objective
# What types of machine learning models perform best on this dataset?
# Which features are most indicative of a poisonous mushroom? 

# Cleanup R Environment ---------------------------------------------------

# Clear all variables and devices (used for plotting) in the environment
rm(list=ls())

# Clear all plots
dev.off()


### ENVIRONMENT SETUP
# *******************************
# Step 1: Set up the environment
# ********************************
# Load the required packages (if packages are not available, install them first)
for (package in c('data.table','caret','readr','ggplot2','ggthemes','dplyr','corrplot','randomForest','H2o')) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package)
    library(package,character.only=T)
  }
}

# ********************************
# Step 2: Import and Read the data
# ********************************

# Load H2o library into R environment
library(h2o)
# Make a connection to the h2o server
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O
h2o.init(ip="localhost", port = 54321, startH2O = TRUE)

# Find and import data into H2O

start <- Sys.time()
mushrooms_csv <- "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
mushrooms.hex <- h2o.importFile(path = mushrooms_csv,destination_frame = "mushrooms_data.hex")
parseTime <- Sys.time() - start
print(paste("Took",round(parseTime, digits = 2),"seconds to parse", nrow(mushrooms.hex), "rows and", ncol(mushrooms.hex),"columns."))
head(mushrooms.hex)

# Name the column names replacing from C1,C2,C3,.......
names(mushrooms.hex) <- c("class","cap.shape","cap.surface","cap.color","bruises", "odor","gill.attachment","gill.spacing","gill.size","gill.color","stalk.shape","stalk.root","stalk.surface.above.ring","stalk.surface.below.ring","stalk.color.above.ring","stalk.color.below.ring", "veil.type","veil.color","ring.number","ring.type","spore.print.color","population", "habitat")
head(mushrooms.hex)
dim(mushrooms.hex)
apply(mushrooms.hex, 2, function(x) length(h2o.unique(x)))
# [1] "Took 2.84 seconds to parse 8124 rows and 23 columns."

# Check for the Mushroom data we imported into the h2o server at http://localhost:54321/flow/index.html 

# ******************************************
# Step 3 : Check the Structure of the Data
# ******************************************

# Take a look at the data
head(mushrooms.hex)
#   C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19 C20 C21 C22 C23
#1  p  x  s  n  t  p  f  c  n   k   e   e   s   s   w   w   p   w   o   p   k   s   u
#2  e  x  s  y  t  a  f  c  b   k   e   c   s   s   w   w   p   w   o   p   n   n   g
#3  e  b  s  w  t  l  f  c  b   n   e   c   s   s   w   w   p   w   o   p   n   n   m
#4  p  x  y  w  t  p  f  c  n   n   e   e   s   s   w   w   p   w   o   p   k   s   u
#5  e  x  s  g  f  n  f  w  b   k   t   e   s   s   w   w   p   w   o   e   n   a   g
#6  e  x  y  y  t  a  f  c  b   n   e   c   s   s   w   w   p   w   o   p   k   n   g

# *****************************************
# Step 4 : Check the Structure of the Data
# ******************************************

# Check the dimentions of the data
dim(mushrooms.hex)
#[1] 8124   23
# mushrooms_data is a 'data.frame' with	8124 obs. of  23 variables

# ******************************************
# Step 5 : Check the Structure of the Data
# ******************************************
str(mushrooms.hex)

# From the structure we can see  all the variables have Factor objects with differnet levels.
# Factors are the r-objects which are created using a vector. It stores the vector along with the distinct values of the elements in the vector as labels. The labels are always character irrespective of whether it is numeric or character or Boolean etc. in the input vector.


# Let us see how many levels each variable have 

h2o.levels(mushrooms.hex)
#h2o.unique(mushrooms.hex)
# apply(mushroom_data, 2, function(x) length(unique(x)))
# We can see the data is represented as a single letter which requirees regular reference to the Attibutes description. We can transform the data with defined attributes.

#**************************************
#Step 6 : Check for missing values NA's
#**************************************
any(is.na(mushrooms.hex))
# [1] 0 - this means no NA's found 

#  1. None of the data are missing the dataset is Structured
#  2. you dont have to deal with omitting rows or columns incase there are most missing values.
#  3. you have accurate and not any predictied or average value replacing the missing data.
#  4. Less time consumption.

#*********************************************
# DATA VISUALIZATION WITH EXPLORATORY ANALYSIS
# ********************************************
#Let us use ggplot2 to visualize the data and get more understanding.
library(ggplot2)
library(ggthemes)
library(dplyr)

#****************
# BUILDING MODELS
#****************

# This function will do the test,train and validation data split and build Random forest,GLM,GBM and Deep Learning Model.

## First, we will create three splits for train/test/valid independent data sets.
## We will train a data set on one set and use the others to test the validity
## The second set will be used for validation most of the time. The third set will
##  be withheld until the end, to ensure that our validation accuracy is consistent
##  with data we have never seen during the iterative process. 


# splits <- function(data){
splits <- h2o.splitFrame(
  mushrooms.hex,         ##  splitting the H2O frame we read above
  ratios = c(0.6,0.2),   ##  create splits of 60% and 20%; 
  ##  H2O will create one more split of 1-(sum of these parameters)
  ##  so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
  seed=1)                ##  setting a seed will ensure reproducible results (not R's seed)
train <- h2o.assign(splits[[1]], "train.hex")   
## assign the first result the R variable train
## and the H2O name train.hex
valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
test <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex

print(paste("Training data has", ncol(train),"columns and", nrow(train), "rows, whereas test data has", nrow(test), "rows, and validation data has rows", nrow(valid))
)

# Take a look at the first few rows of the data set
train[1:5,]   ## rows 1-5, all columns


myY <- "class"
myX <- setdiff(names(train), myY)

## Run our first predictive model (Random Forest Model)
mush_rf_model  <- h2o.randomForest(x = myX,
                 y = myY,
                 training_frame = train,
                 validation_frame = test,
                 model_id = "mush_rf_model",
                 ntrees = 250,
                 max_depth = 30,
                 seed = 1)
print(mush_rf_model)
#Let us see what variables are important in this model.
h2o.varimp_plot(mush_rf_model, num_of_features = NULL)
h2o.confusionMatrix(mush_rf_model)

## Run GBM

mush_gbm_model <- h2o.gbm(x=myX,build_tree_one_node = T,
            y = myY,
            training_frame = train,
            validation_frame = test,
            model_id = "mush_gbm_model",
            ntrees = 500,
            max_depth = 6,
            learn_rate = 0.1)


# Print model performance using train data
print(mush_gbm_model)
h2o.varimp_plot(mush_gbm_model, num_of_features = NULL)
h2o.confusionMatrix(mush_gbm_model)

## Run GLM 
mush_glm_model <- h2o.glm(x = myX,
                          y = myY,
                          training_frame = train,
                          validation_frame = test,
                          lambda = 0,
                          family = "binomial")
                          
print(mush_glm_model)
h2o.std_coef_plot(mush_glm_model, num_of_features = NULL)
# 

## Run DeepLearning
mush_dl_model <- h2o.deeplearning(x = myX,
                        y = myY,
                        training_frame = train,
                        validation_frame = test,
                        activation = "TanhWithDropout",
                        input_dropout_ratio = 0.2,
                        hidden_dropout_ratios = c(0.5,0.5,0.5),
                        hidden = c(50,50,50),
                        epochs = 100,
                        seed = 123456)
print(mush_dl_model)

plot(mush_dl_model)

## Performance on validation set
h2o.confusionMatrix(mush_dl_model)

# note Warning message:
#In .h2o.startModelJob(algo, params, h2oRestApiVersion) :
# Dropping bad and constant columns: [veil.type].


## PART 2 OF THE MUSHROOM DATASET ANALYSIS

# Subset the data using the top 10 features.
mushrooms.hex1 <- mushrooms.hex[1:10, c(1,6,9,10,12,13,20,21,22,23)]
mushrooms.hex1
names(mushrooms.hex1) <- c("class","odor","gill.size","gill.color","stalk.root","stalk-surface-above-ring","ring.type","spore.print.color","population","Habitat")
head(mushrooms.hex1)

# Data Transformation
# Transforms the Class column
class_trans <- function(key){
  switch (key,
          'p' = 'poisonous',
          'e' = 'edible'
  )
}

#Transforms the odor column
odor_trans <- function(key)(
  switch(key,
         'a' = 'almond',
         'l' = 'anise',
         'c'= 'creosote',
         'y'= 'fishy',
         'f'= 'foul',
         'm'= 'musty',
         'n'= 'none',
         'p'= 'pungent',
         's'= 'spicy'
  )
)


# Transforms the gill.size column
gill.size_trans <- function(key){
  switch(key,
         'b'= 'broad',
         'n'= 'narrow')}


#Transforms the gill.color column
gill.color_trans <- function(key){
  switch(key,
         'k'= 'black',
         'n'= 'brown',
         'b'= 'buff',
         'h'= 'chocolate',
         'g'= 'gray')}



#Transforms the stalk.root column
stalk.root_trans <- function(key){
  switch(key,
         'b'= 'bulbous',
         'c'= 'club',
         'u'= 'cup',
         'e'= 'equal',
         'z'= 'rhizomorphs',
         'r'= 'rooted',
         '?'= 'missing')}

#Transforms the stalk.surface.above.ring column
stalk.surface.above.ring_trans <- function(key){
  switch(key,
         'f'= 'fibrous',
         'y'= 'scaly',
         'k'= 'silky',
         's'= 'smooth')}


# Transforms the ring.type column
ring.type_trans <- function(key){
  switch(key,
         'c' = 'cobwebby',
         'e' = 'evanescent',
         'f' = 'flaring',
         'l' = 'large',
         'n' = 'none',
         'p' = 'pendant',
         's' = 'sheathing',
         'z' = 'zone')}

#Transforms the spore.print.color column
spore.print.color_trans <- function(key){
  switch(key,
         'k'=  'black',
         'n'=  'brown',
         'b' = 'buff',
         'h'=  'chocolate',
         'r' = 'green',
         'o' = 'orange',
         'u' = 'purple',
         'w' = 'white',
         'y' = 'yellow')}

#Transforms the population column
population_trans <- function(key){
  switch(key, 
         'a' = 'abundant',
         'c' = 'clustered',
         'n' = 'numerous',
         's' = 'scattered',
         'v' = 'several',
         'y' = 'solitary')}

# Transforms the habitat column
habitat_trans <- function(key){
  switch(key,
         'g' = 'grasses',
         'l' = 'leaves',
         'm' = 'meadows',
         'p' = 'paths',
         'u' = 'urban',
         'w' = 'waster',
         'd' = 'woods')}


# Applying data transformation on the mushroom dataset

mushrooms.hex1$class <- sapply(mushrooms.hex1$class,class_trans)
mushrooms.hex1$`spore.print.color` <- sapply(mushrooms.hex1$`spore.print.color`,spore.print.color_trans)
mushrooms.hex1$`gill.color` <- sapply(mushrooms.hex1$`gill.color`,gill.color_trans)
mushrooms.hex1$`stalk-surface-above-ring` <- sapply(mushrooms.hex1$`stalk-surface-above-ring`,stalk-surface-above-ring_trans)
mushrooms.hex1$`gill.size` <- sapply(mushrooms.hex1$`gill.size`,gill.color_trans) 
mushrooms.hex1$`stalk.root` <- sapply(mushrooms.hex1$`stalk.root`,stalk.root_trans)
mushrooms.hex1$`ring.type` <- sapply(mushrooms.hex1$`ring.type`, ring.type_trans)
mushrooms.hex1$`odor` <- sapply(mushrooms.hex1$`odor`,odor_trans)
mushrooms.hex1$`population` <- sapply(mushrooms.hex1$`population`,population_trans)
mushrooms.hex1$`habitat` <- sapply(mushrooms.hex1$`habitat`, habitat_trans)

mushrooms.hex1


mushroom_features <-  lapply(seq(from=2, to=ncol(mushrooms.hex)), 
                             function(x) {table(mushrooms.hex$class, mushrooms.hex[,x])})
names(mushroom_features) <- colnames(mushrooms.hex)[2:ncol(mushrooms.hex)]
for(i in 1:length(mushroom_features)) {
  print("=======================")
  print(names(mushroom_features)[i])
  print(mushroom_features[[i]]) 
}

h2o.shutdown(prompt=FALSE)
