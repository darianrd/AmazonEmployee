library(tidymodels)
library(themis)
library(vroom)
library(discrim)
library(embed)
library(lme4)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_smote(all_outcomes(), neighbors = 5) |> 
  step_downsample()

prepped_recipe <- prep(amazon_recipe)
baked_recipe <- bake(prepped_recipe, new_data = amazontrain)



## Logistic Regression

# Create logistic regression model
logregmod <- logistic_reg() |> # Type of model
  set_engine("glm") # Fit generalized linear model

# Create workflow
logregworkflow <- workflow() |> 
  add_model(logregmod) |> 
  add_recipe(amazon_recipe) |> 
  fit(data = amazontrain)

# Make predictions
logpreds <- logregworkflow |>
  predict(new_data = amazontest, type = "prob") # Output predictions

# Prep for Kaggle submission
kaggle_sub <- logpreds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./SMOTELogPreds.csv", delim = ",")



## Penalized Logistic Regression

# Create penalized logistic model
penlog_mod <- logistic_reg(mixture = tune(),
                           penalty = tune()) |> # Type of model
  set_engine("glmnet")

# Create workflow
penlog_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(penlog_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV
CV_results <- penlog_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit
final_penlog_workflow <- penlog_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
penlog_preds <- final_penlog_workflow |>
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission
kaggle_sub <- penlog_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./SMOTEPenLogPreds.csv", delim = ",")



## Random Forest

# Create random forest model
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) |> 
  set_engine("ranger") |> 
  set_mode("classification")

# Create workflow
rf_workflow <- workflow() |> 
  add_model(rf_mod) |> 
  add_recipe(amazon_recipe)

# Grid of values to tune over
tuning_grid <- grid_regular(mtry(range=c(1,20)),
                            min_n(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV
CV_results <- rf_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit
final_rf_workflow <- rf_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
rf_preds <- final_rf_workflow |>
  predict(new_data = amazontest, type = "prob") # Output predictions

# Prep for Kaggle submission
kaggle_sub <- rf_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./SMOTERFPreds.csv", delim = ",")



## K-Nearest Neighbors

# Create K-nearest neighbors model
knn_mod <- nearest_neighbor(neighbors = tune()) |> 
  set_mode("classification") |> 
  set_engine("kknn")

# Create workflow
knn_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(knn_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV
CV_results <- knn_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit
final_knn_workflow <- knn_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
knn_preds <- final_knn_workflow |> 
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission
kaggle_sub <- knn_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./SMOTEKNNPreds.csv", delim = ",")



## Naive Bayes

# Create naive Bayes model
nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

# Create workflow
nb_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(nb_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV
CV_results <- nb_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit
final_nb_workflow <- nb_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
nb_preds <- final_nb_workflow |>
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission
kaggle_sub <- nb_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./SMOTENBPreds.csv", delim = ",")