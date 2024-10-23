library(tidymodels)
library(vroom)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> # Target encoding
  step_normalize(all_numeric_predictors()) # Make mean = 0, SD = 1

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
final_workflow <- rf_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Make predictions
rf_preds <- final_workflow |>
  predict(new_data = amazontest, type = "prob") # Output predictions

# Prep for Kaggle submission
kaggle_sub <- rf_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./RFPreds.csv", delim = ",")