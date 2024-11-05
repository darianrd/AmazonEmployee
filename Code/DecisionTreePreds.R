library(tidymodels)
library(vroom)
library(embed)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# Create decision tree model
dtree_mod <- decision_tree(cost_complexity = tune(),
                          tree_depth = tune(),
                          min_n = tune()) |> 
  set_mode("classification") |> 
  set_engine("rpart")

# Create workflow
dtree_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |> 
  add_model(dtree_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(cost_complexity(),
                            tree_depth(),
                            min_n(),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run CV
CV_results <- dtree_workflow |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tune <- CV_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow
final_workflow <- dtree_workflow |> 
  finalize_workflow(best_tune) |> 
  fit(data = amazontrain)

# Get predictions
dtree_preds <- final_workflow |> 
  predict(new_data = amazontest)

# Prep for Kaggle submission
kaggle_sub <- dtree_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_class) |> # Rename .pred_class to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./DecisionTreePreds.csv", delim = ",")
