library(tidyverse)
library(gridExtra)

full_model <- read.csv("./playground/disagree/data/full_model.csv") %>%
  (function(d) {
     d$trial <- seq_len(nrow(d))
     return(d)
})
same_lr <- read.csv("./playground/disagree/data/same_lr.csv") %>%
  (function(d) {
     d$trial <- seq_len(nrow(d))
     return(d)
})
same_preds <- read.csv("./playground/disagree/data/same_initpreds.csv") %>%
  (function(d) {
     d$trial <- seq_len(nrow(d))
     return(d)
})


full_pred <- ggplot(data = full_model) +
  geom_line(aes(x = trial, y = prob_0), linetype = "dotted") +
  geom_line(aes(x = trial, y = prob_1), color = "red", linetype = "dashed") +
  geom_line(aes(x = trial, y = pred_0)) +
  geom_line(aes(x = trial, y = pred_1), color = "red") +
  theme_bw() +
  theme(aspect.ratio = 0.5)

full_curiosity <- ggplot(data = full_model) +
  geom_line(aes(x = trial, y = disagreement_0)) +
  geom_line(aes(x = trial, y = disagreement_1), color = "red") +
  geom_vline(xintercept = 500, color = "red", linetype = "dashed") +
  geom_vline(xintercept = 1000, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "red", linetype = "dashed") +
  theme_bw() +
  theme(aspect.ratio = 0.5)


same_lr_pred <- ggplot(data = same_lr) +
  geom_line(aes(x = trial, y = prob_0), linetype = "dashed") +
  geom_line(aes(x = trial, y = prob_1), color = "red", linetype = "dashed") +
  geom_line(aes(x = trial, y = pred_0)) +
  geom_line(aes(x = trial, y = pred_1), color = "red") +
  theme_bw() +
  theme(aspect.ratio = 0.5)

same_lr_curiosity <- ggplot(data = same_lr) +
  geom_line(aes(x = trial, y = disagreement_0)) +
  geom_line(aes(x = trial, y = disagreement_1), color = "red") +
  geom_vline(xintercept = 500, color = "red", linetype = "dashed") +
  geom_vline(xintercept = 1000, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "red", linetype = "dashed") +
  theme_bw() +
  theme(aspect.ratio = 0.5)


same_preds_pred <- ggplot(data = same_preds) +
  geom_line(aes(x = trial, y = prob_0), linetype = "dashed") +
  geom_line(aes(x = trial, y = prob_1), color = "red", linetype = "dashed") +
  geom_line(aes(x = trial, y = pred_0)) +
  geom_line(aes(x = trial, y = pred_1), color = "red") +
  theme_bw() +
  theme(aspect.ratio = 0.5)

same_preds_curiosity <- ggplot(data = same_preds) +
  geom_line(aes(x = trial, y = disagreement_0)) +
  geom_line(aes(x = trial, y = disagreement_1), color = "red") +
  geom_vline(xintercept = 500, color = "red", linetype = "dashed") +
  geom_vline(xintercept = 1000, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "black", linetype = "dotted") +
  geom_vline(xintercept = 1500, color = "red", linetype = "dashed") +
  theme_bw() +
  theme(aspect.ratio = 0.5)

grid.arrange(full_pred, same_lr_pred, same_preds_pred,
             full_curiosity, same_lr_curiosity, same_preds_curiosity,
             nrow = 2)
