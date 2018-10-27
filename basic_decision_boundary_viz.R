#
# visualize decision boundaries for various clasifiers
#
# original code from:
# http://michael.hahsler.net/SMU/EMIS7332/R/viz_classifier.html
# credit Michael Hahsler
#
# load packages
#
# caret for knn model
#
  library(caret)
#  
# e1017 for naieve bayes model and svm
#  
  library(e1071)
#
# randomForest for randdom Forest classifier
#
  library(randomForest)
#
# MASS for linear discriminant analysis
#
  library(MASS)
#
# rpart for recursive partiioning and regression trees
#
  library(rpart)
#
# nnet for simple 1-hidden layer neural nets
#
  library(nnet)
#  
  decisionplot <- function(model, data, class = NULL, predict_type = "class",
                           resolution = 100, showgrid = TRUE, ...) {
    
    if(!is.null(class)) cl <- data[, class] else cl <- 1
    data <- data[, 1:2]
    k <- length(unique(cl))
    
    plot(data, col = as.integer(cl) + 1L, pch = as.integer(cl) + 1L, ...)
#    
# make grid
#    
    r <- sapply(data, range, na.rm = TRUE)
    xs <- seq(r[1, 1], r[2, 1], length.out = resolution)
    ys <- seq(r[1, 2], r[2, 2], length.out = resolution)
    g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
    colnames(g) <- colnames(r)
    g <- as.data.frame(g)
#    
# guess how to get class labels from predict
# (unfortunately not very consistent between models)
#    
    p <- predict(model, g, type = predict_type)
    if(is.list(p)) p <- p$class
    p <- as.factor(p)
#    
    if(showgrid) points(g, col = as.integer(p)+1L, pch = ".")
#    
    z <- matrix(as.integer(p), nrow = resolution, byrow = TRUE)
    contour(xs, ys, z, add = TRUE, drawlabels = FALSE,
            lwd = 2, levels = (1:(k - 1)) + 0.5)
#    
    invisible(z)
  }
#
  set.seed(1000)
#
# get some data
#  
  data(iris)
#
# Two class case
#  x <- iris[1:100, c("Sepal.Length", "Sepal.Width", "Species")]
#  x$Species <- factor(x$Species)
#
# Three classes
#  
  x <- iris[1:150, c("Sepal.Length", "Sepal.Width", "Species")]
#
# Easier to separate
#  x <- iris[1:150, c("Petal.Length", "Petal.Width", "Species")]
#
  model <- knn3(Species ~ ., data = x, k = 1)
  decisionplot(model, x, class = "Species", main = "kNN with 1 neighbor")
#  
  model <- knn3(Species ~ ., data = x, k = 10)
  decisionplot(model, x, class = "Species", main = "kNN with 10 neighbors")
#  
  model <- naiveBayes(Species ~ ., data = x)
  decisionplot(model, x, class = "Species", main = "naive Bayes")
#  
  model <- lda(Species ~ ., data = x)
  decisionplot(model, x, class = "Species", main = "LDA")
#  
  model <- glm(Species ~., data = x, family = binomial(link = 'logit'))
  class(model) <- c("lr", class(model))
  predict.lr <- function(object, newdata, ...) {
    predict.glm(object, newdata, type = "response") > .5
  }
  decisionplot(model, x, class = "Species", main = "Logistic Regression")
#  
  model <- rpart(Species ~ ., data = x)
  decisionplot(model, x, class = "Species", main = "CART")
#  
  model <- rpart(Species ~ ., data = x,
                 control = rpart.control(cp = 0.001, minsplit = 1))
  decisionplot(model, x, class = "Species", main = "CART (overfitting)")
#  
  model <- randomForest(Species ~ ., data = x)
  decisionplot(model, x, class = "Species", main = "Random Forest")
#  
  model <- svm(Species ~ ., data = x, kernel="linear")
  decisionplot(model, x, class = "Species", main = "SVM (linear kernel)")
#  
  model <- svm(Species ~ ., data = x, kernel = "radial")
  decisionplot(model, x, class = "Species", main = "SVM (radial kernel)")
#  
  model <- svm(Species ~ ., data = x, kernel = "polynomial")
  decisionplot(model, x, class = "Species", main = "SVM (polynomial kernel)")
#  
  model <- svm(Species ~ ., data = x, kernel = "sigmoid")
  decisionplot(model, x, class = "Species", main = "SVM (sigmoid kernel)")
#  
  model <- nnet(Species ~ ., data = x, size = 1, maxit = 1000, trace = FALSE)
  decisionplot(model, x, class = "Species", main = "NN w/ 1 unit in one hidden layer")
#  
  model <- nnet(Species ~ ., data = x, size = 2, maxit = 1000, trace = FALSE)
  decisionplot(model, x, class = "Species", main = "NN w/ 2 units in one hidden layer")
#  
  model <- nnet(Species ~ ., data = x, size = 4, maxit = 1000, trace = FALSE)
  decisionplot(model, x, class = "Species", main = "NN w/ 4 units in one hidden layer")
#  
  model <- nnet(Species ~ ., data = x, size = 10, maxit = 1000, trace = FALSE)
  decisionplot(model, x, class = "Species", main = "NN w/ 10 units in one hidden layer")
  