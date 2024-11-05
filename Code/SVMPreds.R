library(tidymodels)
library(vroom)
library(discrim)
library(embed)
library(lme4)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> # Target encoding
  step_normalize(all_predictors()) |> 
  step_pca(all_predictors(), threshold = 0.8)

# Create SVM models
svmPoly <- svm_poly(degree = tune(),
                    cost = tune()) |> 
  set_mode("classification") |> 
  set_engine("kernlab")

svmRad <- svm_rbf(rbf_sigma = tune(),
                  cost = tune()) |> 
  set_mode("classification") |> 
  set_engine("kernlab")

svmLin <- svm_linear(cost = tune()) |> 
  set_mode("classification") |> 
  set_engine("kernlab")

# Create SVM polynomial workflow
poly_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(svmPoly)

# Grid of values to tune over (polynomial)
poly_tuning_grid <- grid_regular(degree(),
                                 cost(),
                                 levels = 5)

# Split data for CV (polynomial)
poly_folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV (polynomial)
poly_CV_results <- poly_workflow |> 
  tune_grid(resamples = poly_folds,
            grid = poly_tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters (polynomial)
poly_best_tune <- poly_CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit (polynomial)
final_poly_workflow <- poly_workflow |> 
  finalize_workflow(poly_best_tune) |> 
  fit(data = amazontrain)

# Make predictions (polynomial)
poly_preds <- final_poly_workflow |>
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission (polynomial)
poly_kaggle_sub <- poly_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file (polynomial)
vroom_write(x = poly_kaggle_sub, file = "./SVMPolyPreds.csv", delim = ",")

# Create SVM radial workflow
rad_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(svmRad)

# Grid of values to tune over (radial)
rad_tuning_grid <- grid_regular(rbf_sigma(),
                                cost(),
                                levels = 5)

# Split data for CV (radial)
rad_folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV (radial)
rad_CV_results <- rad_workflow |> 
  tune_grid(resamples = rad_folds,
            grid = rad_tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters (radial)
rad_best_tune <- rad_CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit (radial)
final_rad_workflow <- rad_workflow |> 
  finalize_workflow(rad_best_tune) |> 
  fit(data = amazontrain)

# Make predictions (radial)
rad_preds <- final_rad_workflow |>
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission (radial)
rad_kaggle_sub <- rad_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file (radial)
vroom_write(x = rad_kaggle_sub, file = "./SVMRadPreds.csv", delim = ",")

# Create SVM linear workflow
lin_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(svmLin)

# Grid of values to tune over (linear)
lin_tuning_grid <- grid_regular(cost(),
                                levels = 5)

# Split data for CV (linear)
lin_folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV (linear)
lin_CV_results <- lin_workflow |> 
  tune_grid(resamples = rad_folds,
            grid = lin_tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters (linear)
lin_best_tune <- lin_CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow and fit (linear)
final_lin_workflow <- lin_workflow |> 
  finalize_workflow(lin_best_tune) |> 
  fit(data = amazontrain)

# Make predictions (linear)
lin_preds <- final_lin_workflow |>
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission (linear)
lin_kaggle_sub <- lin_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file (linear)
vroom_write(x = lin_kaggle_sub, file = "./SVMLinPreds.csv", delim = ",")
