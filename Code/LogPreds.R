library(tidymodels)
library(vroom)

amazontrain <- vroom("train.csv")
amazontest <- vroom("test.csv")

amazontrain$ACTION <- factor(amazontrain$ACTION) # Make ACTION (response) a factor

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazontrain) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = 0.001) |> # Combine categorical values that occur 0.1% of the time
  step_dummy(all_nominal_predictors()) # Create dummy variables

# Create logistic regression model
logregmod <- logistic_reg() |> # Type of model
  set_engine("glm") # Fit generalized linear model

# Create workflow
logregworkflow <- workflow() |> 
  add_model(logregmod) |> 
  add_recipe(amazon_recipe) |> 
  fit(data = amazontrain)

# Make predictions
logpreds <- predict(logregworkflow,
                    new_data = amazontest,
                    type = "prob") # Output predictions

# Prep for Kaggle submission
kaggle_sub <- logpreds %>%
  bind_cols(., amazontest) |> # Bind predictions to test data
  rename(ACTION = .pred_1) |> # Rename .pred_1 to ACTION for Kaggle submission
  select(id, ACTION) # Keep id and ACTION variables

# Write out file
vroom_write(x = kaggle_sub, file = "./LogPreds.csv", delim = ",")
