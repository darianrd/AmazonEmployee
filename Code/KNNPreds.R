library(tidymodels)
library(vroom)
library(embed)
library(kknn)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> # Target encoding
  step_normalize(all_numeric_predictors()) # Make mean = 0, SD = 1

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
final_workflow <- knn_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
knn_preds <- final_workflow |> 
  predict(new_data = amazontest, type = "prob")

# Prep for Kaggle submission
kaggle_sub <- knn_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./KNNPreds.csv", delim = ",")
