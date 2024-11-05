library(tidymodels)
library(vroom)
library(parsnip)
library(tidyselect)
library(dbarts)
library(lme4)
library(embed)

# Read in data
amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# Create BART model
bart_mod <- parsnip::bart(trees = tune()) |> 
  set_mode("classification") |> 
  set_engine("dbarts")

# Create workflow for BART model
bart_workflow <- workflow() |> 
  add_recipe(amazon_recipe) |>
  add_model(bart_mod)

# Grid of values to tune over
grid_tuning <- grid_regular(trees(),
                            levels = 5)

# Split data for cross-validation
folds <- vfold_cv(amazontrain, v = 5, repeats = 1)

# Run cross-validation
bart_CV <- bart_workflow |> 
  tune_grid(resamples = folds,
            grid = grid_tuning,
            metrics = metric_set(roc_auc))

# Find best tuning parameters
best_tuning <- bart_CV |> 
  select_best(metric = "roc_auc")

# Finalize workflow
final_workflow <- bart_workflow |> 
  finalize_workflow(best_tuning) |> 
  fit(data = amazontrain)

# Get predictions
bart_preds <- final_workflow |> 
  predict(new_data = amazontest)

# Prep for Kaggle submission
kaggle_sub <- bart_preds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_class) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./BARTPreds.csv", delim = ",")
