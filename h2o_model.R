
library(h2o)
h2o.init()
h2o.shutdown()


# import the dataset
df <- h2o.importFile("C:\\Users\\coneill\\Documents\\IndependentStudy\\Independent_Study_Data_gather\\Data\\usgs_gages2.csv")

# convert columns to factors

# if you have a column with integers that represents a class in a classification problem 
# (0, 1), you will have to change the column type from numeric to categorical/factor 

df$land_cover_value <- as.factor(df$land_cover_value)
df$soil_value <- as.factor(df$soil_value)
df$flood <- as.factor(df$flood)


# set the predictor and response columns
predictors <- c("alt_va", "drain_area", "aspect_value", "dem_value", 
                "land_cover_value", "ndmi_value", "ndvi_value",
                "precip_5yr_value", "precip_10yr_value", "precip_25yr_value",
                "slope_value", "soil_value", "gage_height.min", "gage_height.max",
                "gage_height.median")
response <- "flood"

# split the dataset into train and test sets
splits <- h2o.splitFrame(data =  df, ratios = c(0.8, 0.1), seed = 1234)
names(splits) <- c("train","valid","test")

train <- splits$train
valid <- splits$valid
test <- splits$test


# Identify predictors and response
x <- predictors
y <- response

nfolds <- 5

###############
## GLM Model ##
###############

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
                     standardize = TRUE) 

# # predict using the GLM model and the testing dataset
# predict_glm <- h2o.predict(object = glm_grid, newdata = test)
# 
# # view a summary of the predictions
# h2o.head(predict_glm)
# predict_glm
# 
# # Basic RMSE value
# rmse_basic_glm <- h2o.rmse(grid_glm)
# rmse_basic_glm

# # RMSE value for both training and validation data
# rmse_flood_glm <- h2o.rmse(glm_grid, train = TRUE, valid = TRUE, xval= FALSE)
# rmse_flood_glm



###############
## GBM Model ##
###############

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
                      stopping_metric = "RMSE",
                      stopping_rounds = 5,
                      stopping_tolerance = 0.05,
                      score_each_iteration = T,
                      seed = 23123,
                      keep_cross_validation_predictions = TRUE,
                      keep_cross_validation_models = TRUE,
                      distribution = "bernoulli")
 
# # predict using the GBM model and the testing dataset
# predict_gbm <- h2o.predict(object = gbm_grid, newdata = test)
# 
# # view a summary of the predictions
# h2o.head(predict_gbm)
# predict_gbm
# 
# # Basic RMSE value
# rmse_basic_gbm <- h2o.rmse(gbm_grid)
# rmse_basic_gbm
# 
# # RMSE value for both training and validation data
# rmse_flood_gbm <- h2o.rmse(gbm_grid, train = TRUE, valid = TRUE, xval= FALSE)
# rmse_flood_gbm
# 
# h2o.gainsLift(gbm_grid, valid=TRUE, xval=FALSE)


###################
## Random Forest ##
###################

rf_grid <- h2o.grid(algorithm = "drf",
                    grid_id = "rf_grid",
                    training_frame = train,
                    validation_frame = valid,
                    x=predictors,
                    y=response,
                    nfolds=nfolds,
                    fold_assignment = "Stratified",
                    ntrees = 200,
                    max_depth = 5,
                    stopping_metric = "RMSE",
                    stopping_rounds = 5,
                    stopping_tolerance = 0.05,
                    score_each_iteration = T,
                    seed = 23123,
                    keep_cross_validation_predictions = TRUE,
                    keep_cross_validation_models = TRUE,
                    min_split_improvement = 0.00001)

# # predict using the Random Forest model and the testing dataset
# predict_rf <- h2o.predict(object = rf_grid, newdata = test)
# 
# # view a summary of the predictions
# h2o.head(predict_rf)
# predict_rf
# 
# # Basic RMSE value
# rmse_basic_rf <- h2o.rmse(rf_grid)
# rmse_basic_rf
# 
# # RMSE value for both training and validation data
# rmse_flood_rf <- h2o.rmse(rf_grid, train = TRUE, valid = TRUE, xval= FALSE)
# rmse_flood_rf
# 
# h2o.gainsLift(rf_grid, valid=TRUE, xval=FALSE)

###################
## Deep Learning ##
###################


dl_grid <- h2o.grid(algorithm = "deeplearning",
                    model_id = "flood_dl",
                    x = predictors,
                    y = response,
                    nfolds=nfolds,
                    adaptive_rate = TRUE,
                    rho = 0.95,
                    epsilon = 1e-5,
                    fold_assignment = "stratified",
                    distribution = "bernoulli",
                    hidden = c(25,25,25,25,25,1),
                    epochs = 50,
                    train_samples_per_iteration = -1,
                    reproducible = TRUE,
                    activation = "Tanh",
                    loss = "Quadratic",
                    l1=1e-5,
                    l2=1e-5,
                    seed = 23123,
                    stopping_metric = "RMSE",
                    stopping_tolerance=1e-4,        ## stop when misclassification does not improve by >=1% for 2 scoring events
                    stopping_rounds=5,
                    training_frame = train,
                    validation_frame = valid,
                    keep_cross_validation_predictions = TRUE,
                    keep_cross_validation_models = TRUE,
                    overwrite_with_best_model = TRUE,
                    standardize = TRUE)
                               

# # predict using the Random Forest model and the testing dataset
# predict_dl <- h2o.predict(object = dl_grid, newdata = test)
# 
# # view a summary of the predictions
# h2o.head(predict_dl)
# predict_dl
# 
# # Basic RMSE value
# rmse_basic_dl <- h2o.rmse(dl_grid)
# rmse_basic_dl
# 
# # RMSE value for both training and validation data
# rmse_flood_dl <- h2o.rmse(dl_grid, train = TRUE, valid = TRUE, xval= FALSE)
# rmse_flood_dl
# 
# h2o.gainsLift(dl_grid, valid=TRUE, xval=FALSE)

######################
## Stacked Ensemble ##
######################

models <- c(gbm_grid, glm_grid, rf_grid, dl_grid)
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = models)
ensemble
gainsLife_ensemble <- h2o.gainsLift(ensemble)
gainsLife_ensemble

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

gainsLift_perf <- h2o.gainsLift(perf)
gainsLift_perf


# Compare to base learner performance on the test set
perf_glm_test <- h2o.performance(h2o.getModel(glm_grid@model_ids[[1]]), newdata = test)
perf_rf_test <- h2o.performance(h2o.getModel(rf_grid@model_ids[[1]]), newdata = test)
perf_gbm_test <- h2o.performance(h2o.getModel(gbm_grid@model_ids[[1]]), newdata = test)
perf_dl_test <- h2o.performance(h2o.getModel(dl_grid@model_ids[[1]]), newdata = test)

baselearner_best_rmse_test <- min(h2o.rmse(perf_glm_test), 
                                  h2o.rmse(perf_rf_test), 
                                  h2o.rmse(perf_gbm_test), 
                                  h2o.rmse(perf_dl_test))

ensemble_rmse_test <- h2o.rmse(perf)
print(sprintf("Best Base-learner Test RMSE:  %s", baselearner_best_rmse_test))
print(sprintf("Ensemble Test RMSE:  %s", ensemble_rmse_test))

# Generate predictions on a test set (if neccessary)
pred <- h2o.predict(ensemble, newdata = test)
pred










