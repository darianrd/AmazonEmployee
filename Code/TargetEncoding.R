library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

amazon_train <- vroom("train.csv")
amazon_test <- vroom("test.csv")

# Create bar chart for ACTION variable
amazon_train |> ggplot(aes(x = ACTION)) +
  geom_bar(fill = "lightblue2",
           color = "lightblue4") +
  scale_y_continuous(expand = expansion(mult = c(0, .01))) +
  labs(x = "Access",
       y = "Count",
       title = "Access Granted to Amazon Employees") +
  theme_classic()

# Create boxplot for ACTION variable and ROLE_CODE variable
amazon_train |> ggplot(aes(x = ACTION, y = ROLE_CODE, group = ACTION)) +
  geom_boxplot(fill = "lightblue2",
               color = "lightblue4") +
  ylim(c(118000, 125000)) +
  stat_boxplot(geom = "errorbar",
               color = "lightblue4",
               width = 0.4) +
  labs(x = "Access",
       title = "Access Granted to Code Role Employees") +
  theme_classic()

# Create recipe
amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> # Turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = 0.001) |> # Combine categorical values that occur
  step_dummy(all_nominal_predictors())
  
prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazon_train)
