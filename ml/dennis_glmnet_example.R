
############################################
### glmnet multinomial example

pacman::p_load(data.table, dplyr, glmnet, yardstick)


############################################
### read in data

### example data with Species prediction and samples in rownames
data(iris)
train_df <- iris
rownames(train_df) <- paste0("sample_", 1:nrow(train_df))


### important to have same factor level everywhere
cancer_labels_levels <- sort(unique(as.character(train_df$Species)))


x_train <- as.matrix(train_df[, -5])  ### predictors df
### target variable
y_train <- factor(train_df$Species, levels = cancer_labels_levels)

table(y_train)

############################################
### functions

get_multinomial_performance <- function(pred_prob, y_true) {
  ### assumes y_true has correct level order
  ### some editing of the prob table to have a suitable format for yardstick -
  ### pred_class is class with max prob
  prob_edited <- data.frame(pred_prob, check.names = FALSE)
  colnames(prob_edited) <- gsub(".s0", "", colnames(prob_edited))
  prob_edited$sample_id <- rownames(prob_edited)
  prob_classes <- prob_edited %>% reshape2::melt(data.table = FALSE) %>%
      group_by(sample_id) %>% dplyr::filter(value == max(value)) %>%
      as.data.frame()
  pred_class <- factor(setNames(prob_classes$variable, prob_classes$sample_id),
                       levels = levels(y_true))
  pred_class <- pred_class[rownames(prob_edited)]
  prob_edited$truth <- y_true
  prob_edited$predicted <- pred_class

  f1_macro <- as.numeric(yardstick::f_meas(prob_edited,
                                           truth = truth,
                                           estimate = predicted,
                                           estimator = "macro")[, ".estimate"])
  f1_micro <- as.numeric(yardstick::f_meas(prob_edited,
                                           truth = truth,
                                           estimate = predicted,
                                           estimator = "micro")[, ".estimate"])
  acc <- as.numeric(yardstick::accuracy(prob_edited,
                                        truth = truth,
                                        estimate = predicted)[, ".estimate"])
  auroc <- as.numeric(yardstick::roc_auc(prob_edited,
                                         truth = truth,
                                         estimator = "hand_till",
                                         levels(y_true))[, ".estimate"])
  auprc <- as.numeric(yardstick::pr_auc(prob_edited,
                                        truth = truth,
                                        estimator = "macro",
                                        levels(y_true))[, ".estimate"])
  mcc <- as.numeric(yardstick::mcc(prob_edited,
                                   truth = truth,
                                   estimate = predicted)[, ".estimate"])

  list("f1_macro" = f1_macro,
       "f1_micro" = f1_micro,
       "acc" = acc,
       "auroc" = auroc,
       "auprc" = auprc,
       "mcc" = mcc)
}


############################################
### perform glmnet

set.seed(123)

REG_ALPHA <- 1   ### lasso
# REG_ALPHA = 0   ### ridge
# REG_ALPHA = 0.5   ### elastic net -
# better to determine using cv with both lambda and alpha


### find best penalization parameter lambda
cv_glmnet <- cv.glmnet(x_train,
                       y_train,
                       alpha = REG_ALPHA,
                       family = "multinomial",
                       intercept = TRUE,
                       standardize = FALSE)
plot(cv_glmnet)

### parallelized version
# require(doMC)
# registerDoMC(cores = 10)
# cv_glmnet = cv.glmnet(x_train, y_train, alpha=REG_ALPHA, family = "multinomial", standardize = T, intercept=T, parallel=T)

REG_LAMBDA <- cv_glmnet$lambda.min
# REG_LAMBDA = cv_glmnet$lambda.1se  ### sometimes gives slightly better performance but not much difference

### create the regularized model with alpha and best lambda values
fit_optimized <- glmnet(x_train,
                        y_train,
                        alpha = REG_ALPHA,
                        lambda = REG_LAMBDA,
                        family = "multinomial",
                        intercept = TRUE,
                        standardize = FALSE)


### predict
train_prob <- predict(fit_optimized, newx = x_train, type = "response")
x <- get_multinomial_performance(pred_prob = train_prob, y_true = y_train)

print(x$acc)
