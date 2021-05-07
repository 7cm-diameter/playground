library(tidyverse)
library(gridExtra)

# read simulated data
add_trial <- function(d) {
  d$trial <- seq_len(nrow(d))
  return(d)
}

add_model <- function(d, model) {
  d$model <- model
  return(d)
}

full_model <- read.csv("./playground/disagree/data/full_model.csv") %>%
  add_trial %>%
  add_model(., "Full model")
same_lr <- read.csv("./playground/disagree/data/same_lr.csv") %>%
  add_trial %>%
  add_model(., "Same learning rate")
same_preds <- read.csv("./playground/disagree/data/same_initpreds.csv") %>%
  add_trial %>%
  add_model(., "Same initial prediction")

alldata <- rbind(full_model, same_lr, same_preds)

# draw figures
leaky_spline <- function(x, lr) {
  ret <- Reduce(function(x1, x2) {
                  (1 - lr) * x1 + lr * x2
          }, x, accumulate = T)
  return(ret)
}

theme_config <- theme_bw() +
                theme(aspect.ratio = 0.5,
                      plot.margin = grid::unit(c(0, 0, 0, 0), "mm"))

show_reward_prob <- function(d) {
  ggplot(data = d) +
    geom_line(aes(x = trial, y = prob_0), size = 1) +
    geom_line(aes(x = trial, y = prob_1), color = "red", size = 1) +
    ylim(0, 1) +
    ylab("Reward probabilities for each arm") +
    theme_config
}

show_prediction <- function(d, lr) {
  ggplot(data = d) +
    geom_line(aes(x = trial, y = pred_0), linetype = "dotted") +
    geom_line(aes(x = trial, y = pred_1), linetype = "dotted", color = "red") +
    geom_line(aes(x = trial, y = leaky_spline(pred_0, lr))) +
    geom_line(aes(x = trial, y = leaky_spline(pred_1, lr)), color = "red") +
    ylab("Average of the predictions of learner for each arm") +
    theme_config
}

show_curiosity <- function(d, lr) {
  ggplot(data = d) +
    geom_line(aes(x = trial, y = disagreement_0), linetype = "dotted") +
    geom_line(aes(x = trial, y = disagreement_1),
              linetype = "dotted", color = "red") +
    geom_line(aes(x = trial, y = leaky_spline(disagreement_0, lr))) +
    geom_line(aes(x = trial, y = leaky_spline(disagreement_1, lr)),
              color = "red") +
    ylab("Variance of the predictions of learner for each arm") +
    theme_config
}

plot_probs <- show_reward_prob(alldata) + facet_wrap(~model)
plot_preds <- show_prediction(alldata, 0.1) + facet_wrap(~model)
plot_curious <- show_curiosity(alldata, 0.1) + facet_wrap(~model)

grid.arrange(plot_probs, plot_preds, plot_curious)
