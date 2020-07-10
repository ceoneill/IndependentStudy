
library(data.table)
library(bit64)
library(dplyr)
library(lubridate)
library(openxlsx)
library(tidyverse)
library(keras)
library(RSNNS)
library(rlang)
#library(lares)
library (readr)
library(h2o)
h2o.init()
#h2o.shutdown()


# import the dataset
df <- read.csv("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages2.csv")
df <- df [ , -1]
# convert columns to factors

# if you have a column with integers that represents a class in a classification problem 
# (0, 1), you will have to change the column type from numeric to categorical/factor 


df_mutate <- df %>% mutate_if(is.character, as.numeric)
df_mutate <- df_mutate %>% mutate_if(is.integer, as.numeric)
df_mutate <- df_mutate %>% mutate_if(is.factor, as.numeric)


#df$land_cover_value <- as.factor(df$land_cover_value)
#df$soil_value <- as.factor(df$soil_value)

#df_mutate$flood <- as.factor(df$flood)


# set the predictor and response columns
predictors <- c("alt_va", "drain_area", "aspect_value", "dem_value", 
                "land_cover_value", "ndmi_value", "ndvi_value",
                "precip_5yr_value", "precip_10yr_value", "precip_25yr_value",
                "slope_value", "soil_value", "gage_height.min", "gage_height.max",
                "gage_height.median")
response <- "flood"

###############
## NORMALIZE ##
###############


df_norm <- normalizeData(df_mutate, type='0_1')
colnames(df_norm) <- names(df)
norm_x <- normalizeData(df_mutate[,predictors], type='0_1')
norm_y <- normalizeData(df_mutate[,response], type='0_1')

df_norm_index <- as.data.frame(df_norm)
df_norm_index$flood <- df$flood
head(df_norm_index)
# convert into H2O frame
df_h2o <- as.h2o(df_norm_index)

df_h2o$flood <- as.factor(df_h2o$flood)

# split the dataset into train and test sets
splits <- h2o.splitFrame(data =  df_h2o, ratios = c(0.8, 0.1), seed = 1234)
names(splits) <- c("train","valid","test")

train <- splits$train
valid <- splits$valid
test <- splits$test


# Identify predictors and response
x <- predictors
y <- response

nfolds <- 5

search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 100,
                        seed = 1)

###############
## GLM Model ##
###############

alpha=seq(0.01, 1, 0.01)
lambda=seq(0.00000001,0.0001, 0.000001)

glm_params = list(alpha = alpha,
                  lambda = lambda)

glm_grid <- h2o.grid(algorithm = "glm",
                     grid_id = "glm_grid", 
                     family="binomial", 
                     x= predictors,
                     y=response,
                     training_frame = train,
                     validation_frame = valid, 
                     nfolds=nfolds,
                     fold_assignment = "Stratified",
                     seed = 23123,
                     early_stopping = TRUE,
                     keep_cross_validation_predictions = TRUE,
                     keep_cross_validation_models = TRUE,
                     standardize = TRUE,
                     hyper_params = glm_params,
                     search_criteria = search_criteria) 


###############
## GBM Model ##
###############

learn_rate = seq(0.01,0.1,0.01)
max_depth = seq(1,10,1)
sample_rate = seq(0.1,1,0.1)
col_sample_rate = seq(0.1, 0.9, 0.01)

gbm_params <- list(learn_rate = learn_rate,
                   max_depth = max_depth,
                   sample_rate = sample_rate,
                   col_sample_rate = col_sample_rate)


gbm_grid <-  h2o.grid(algorithm = "gbm",
                      grid_id = "gbm_grid",
                      x = predictors,
                      y = response,
                      training_frame = train,
                      nfolds=nfolds,
                      fold_assignment = "Stratified",
                      ntrees = 200,
                      max_depth = 5,
                      validation_frame = valid,
                      stopping_metric = "AUC",
                      stopping_rounds = 5,
                      stopping_tolerance = 0.05,
                      score_each_iteration = T,
                      seed = 23123,
                      keep_cross_validation_predictions = TRUE,
                      keep_cross_validation_models = TRUE,
                      distribution = "bernoulli",
                      hyper_params = gbm_params,
                      search_criteria = search_criteria)
 


###################
## Random Forest ##
###################

ntrees = seq(1,100,10)
max_depth = seq(1,100,10)

drf_params = list(ntrees = ntrees, 
                  max_depth = max_depth)

drf_grid <- h2o.grid(algorithm = "drf",
                    grid_id = "drf_grid",
                    training_frame = train,
                    validation_frame = valid,
                    x=predictors,
                    y=response,
                    nfolds=nfolds,
                    fold_assignment = "Stratified",
                    stopping_metric = "AUC",
                    stopping_rounds = 5,
                    stopping_tolerance = 0.05,
                    score_each_iteration = T,
                    seed = 23123,
                    keep_cross_validation_predictions = TRUE,
                    keep_cross_validation_models = TRUE,
                    min_split_improvement = 0.00001,
                    hyper_params = drf_params,
                    search_criteria = search_criteria)


###################
## Deep Learning ##
###################

activation = c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout")
hidden = list(c(5, 5, 5, 5, 5), c(10, 10, 10, 10), c(50, 50, 50), c(100, 100, 100))
epochs = c(50, 100, 200)
l1 = seq(0, 0.00001, 0.0001)
l2 = seq(0, 0.00001, 0.0001)
rate = c(0, 0.1, 0.01)
rate_annealing = c(1e-8, 1e-7, 1e-6)
rho = seq(0.9, 0.999)
epsilon = c(1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
momentum_start = c(0, 0.5)
momentum_stable = c(0.99, 0.5, 0)
input_dropout_ratio = seq(0, 0.5, 0.1)
max_w2 = c(10, 100, 1000, 1000000)

dl_params = list(
  activation = activation, 
  hidden = hidden,
  epochs = epochs,
  l1 = l1, 
  l2 = l2,
  rate = rate,
  rate_annealing = rate_annealing,
  rho = rho,
  epsilon = epsilon,
  momentum_start = momentum_start,
  momentum_stable = momentum_stable,
  input_dropout_ratio = input_dropout_ratio,
  max_w2 = max_w2
)


dl_grid <- h2o.grid(algorithm = "deeplearning",
                    model_id = "flood_dl",
                    grid_id = "dl_grid",
                    x = predictors,
                    y = response,
                    nfolds=nfolds,
                    adaptive_rate = TRUE,
                    fold_assignment = "stratified",
                    distribution = "bernoulli",
                    train_samples_per_iteration = -1,
                    reproducible = TRUE,
                    loss = "Quadratic",
                    seed = 23123,
                    stopping_metric = "AUC",
                    stopping_tolerance=1e-4,        ## stop when misclassification does not improve by >=1% for 2 scoring events
                    stopping_rounds=5,
                    training_frame = train,
                    validation_frame = valid,
                    keep_cross_validation_predictions = TRUE,
                    keep_cross_validation_models = TRUE,
                    overwrite_with_best_model = TRUE,
                    standardize = TRUE,
                    hyper_params = dl_params,
                    search_criteria = search_criteria)
                               



######################
## Stacked Ensemble ##
######################

glm_gridperf <- h2o.getGrid(grid_id = "glm_grid",
                            sort_by = "auc",
                            decreasing = TRUE)

gbm_gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                            sort_by = "auc",
                            decreasing = TRUE)

drf_gridperf <- h2o.getGrid(grid_id = "drf_grid",
                            sort_by = "auc",
                            decreasing = TRUE)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid",
                           sort_by = "auc",
                           decreasing = TRUE)
print(glm_gridperf)
print(gbm_gridperf)
print(drf_gridperf)
print(dl_gridperf)


# Grab the top GLM model, chosen by AUC
best_glm <- h2o.getModel(glm_gridperf@model_ids[[1]])
best_glm

# Grab the top GBM model, chosen by AUC
best_gbm <- h2o.getModel(gbm_gridperf@model_ids[[1]])
best_gbm

# Grab the top DRF model, chosen by AUC
best_drf <- h2o.getModel(drf_gridperf@model_ids[[1]])
best_drf


models <- c(gbm_grid, glm_grid, drf_grid, dl_grid)

ensemble <- h2o.stackedEnsemble(x = x,
                               y = y,
                                 training_frame = train,
                                 base_models = models)
ensemble


# Train a stacked ensemble using the GBM grid
glm_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "glm",
                                    training_frame = train,
                                    base_models = models)

gbm_ensemble <- h2o.stackedEnsemble(x = x,
                                    y = y,
                                    metalearner_algorithm = "gbm",
                                    training_frame = train,
                                    base_models = models)


 drf_ensemble <- h2o.stackedEnsemble(x = x,
                                     y = y,
                                     metalearner_algorithm = "drf",
                                     training_frame = train,
                                     base_models = models)

dl_ensemble <- h2o.stackedEnsemble(x = x,
                                   y = y,
                                   metalearner_algorithm = "deeplearning",
                                   training_frame = train,
                                   base_models = models)


# Compare to base learner performance on the test set
glm_ensemble_test <- h2o.performance(glm_ensemble, newdata = test)
gbm_ensemble_test <- h2o.performance(gbm_ensemble, newdata = test)
drf_ensemble_test <- h2o.performance(drf_ensemble, newdata = test)
dl_ensemble_test <- h2o.performance(dl_ensemble, newdata = test)


# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

# Compare to base learner performance on the test set
perf_glm_test <- h2o.performance(h2o.getModel(glm_grid@model_ids[[1]]), newdata = test)
perf_drf_test <- h2o.performance(h2o.getModel(drf_grid@model_ids[[1]]), newdata = test)
perf_gbm_test <- h2o.performance(h2o.getModel(gbm_grid@model_ids[[1]]), newdata = test)
perf_dl_test <- h2o.performance(h2o.getModel(dl_grid@model_ids[[1]]), newdata = test)

baselearner_best_auc_test <- min(h2o.auc(perf_glm_test), 
                                  h2o.auc(perf_drf_test), 
                                  h2o.auc(perf_gbm_test), 
                                  h2o.auc(perf_dl_test))

ensemble_auc_test <- h2o.auc(dl_ensemble_test)
print(sprintf("Best Base-learner Test auc:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test auc:  %s", ensemble_auc_test))


# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)
pred

######################
## Confusion Matrix ##
######################

#glm_grid_cm <- h2o.getGrid(grid_id = glm_grid, sort_by = "auc", decreasing = TRUE)
glm_cm <- h2o.getModel(glm_grid@model_ids[[1]])
gbm_cm <- h2o.getModel(gbm_grid@model_ids[[1]])
drf_cm <- h2o.getModel(drf_grid@model_ids[[1]])
dl_cm <- h2o.getModel(dl_grid@model_ids[[1]])

confusion <- h2o.confusionMatrix(ensemble, newdata = test)
confusion


#####################
## Mapping Results ##
#####################

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(dl_ensemble, newdata = test)
pred

summary(ensemble)


# pred_ensemble <- as.data.frame(denormalizeData(pred, getNormParameters(norm_y)))
  # have pred_ensemble as a dataframe w/o denormalization 
pred_ensemble <- as.data.frame(pred)

pred_ensemble <- pred_ensemble %>% mutate_if(is.character, as.numeric)
pred_ensemble <- pred_ensemble %>% mutate_if(is.factor, as.numeric)
pred_ensemble <- pred_ensemble %>% mutate_if(is.integer, as.numeric)


test$flood <- as.numeric(test$flood)
test_backtransform <- round(as.data.frame(denormalizeData(test$flood, getNormParameters(norm_y))), 0)

test_df <- as.data.frame(test)


results <- as.data.frame(cbind(test_backtransform, round(pred_ensemble,0)))
results <- cbind(results, test_df$site_no)
colnames(results) <- c("flood", "predicted", "p0", "p1", "site_no")
head(results)

df_test_observed <- df[,c("flood","site_no")]

model_results <- results %>% left_join(df_test_observed, by="site_no")
colnames(model_results) <- c("flood", "predicted", "p0", "p1", "site_no", "?")
head(model_results)

plot(results$flood, results$predicted, xlim=c(0,50000))
abline(x=y,col="blue")

h2o.varimp_plot(h2o.getModel(glm_grid@model_ids[[1]]))
h2o.varimp_plot(h2o.getModel(gbm_grid@model_ids[[1]]))
h2o.varimp_plot(h2o.getModel(drf_grid@model_ids[[1]]))
h2o.varimp_plot(h2o.getModel(dl_grid@model_ids[[1]]))

# lares::mplot_full(tag = results$claims,
#                   score = results$predicted,
#                   splits = 10,
#                   subtitle = "Ensemble DL Metalearner Results",
#                   model_name = "simple_model_02",
#                   save = T)










