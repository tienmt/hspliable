#' Gibbs sampler for Pliable Horseshoe for sparse interaction effects with Missing Data
#'
#' This function performs Bayesian inference for the **pliable lasso model**
#' in the presence of missing response values. The pliable lasso extends
#' the standard lasso by allowing **coefficients of predictors to vary
#' as a linear function of modifying covariates** (Z). In other words, the
#' effect of X on y can be modified by Z:
#'
#' \deqn{y_i = \beta_0 + Z_i \theta_0 + X_i \beta + \sum_j X_{ij} (Z_i^\top \theta_j) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)}
#'
#' where missing entries in y are imputed during the Gibbs sampler.
#'
#' The regression coefficients (\eqn{\beta} for main effects and \eqn{\theta} for modifier interactions)
#' follow **hierarchical Horseshoe priors**, which allow for strong shrinkage of irrelevant
#' predictors while leaving relevant predictors relatively unshrunk:
#'
#' \deqn{\beta_j, \theta_j \mid \lambda_j, \tau \sim N(0, \tau^2 \lambda_j^2 I),}
#' \deqn{\lambda_j^2 \sim \text{Inv-Gamma}(1/2, 1/ \nu_j), \quad \nu_j \sim \text{Inv-Gamma}(1/2, 1),}
#' \deqn{\tau^2 \sim \text{Inv-Gamma}(1/2, 1/\xi), \quad \xi \sim \text{Inv-Gamma}(1/2, 1).}
#'
#' This hierarchical structure is implemented in C++ using **Rcpp and Armadillo** for efficiency.
#'
#' @author The Tien Mai, \email{the.tien.mai@@fhi.no}
#' @references
#' - Tibshirani, R., & Friedman, J. (2020). A pliable lasso. *Journal of Computational and Graphical Statistics, 29*(1), 215-225.
#' - Mai. T.T. (2025). Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses. arXiv
#'
#' @param y Numeric vector of responses of n samples;
#'   may contain \code{NA} for missing values.
#' @param X Numeric matrix of predictors (n x p).
#' @param Z Numeric matrix of modifying covariates (n x q).
#' @param n_iter Number of Gibbs iterations (default 2000).
#' @param burn_in Number of burn-in iterations (default 1000).
#' @param a0,b0 Hyperparameters for the inverse-gamma prior on \eqn{\sigma^2} (default 0.01 each).
#' @param sigma0_sq Prior variance for the intercept block (\eqn{\beta_0, \theta_0}) (default 1.0).
#' @param eps Small ridge term added for numerical stability (default 1e-6).
#' @param verbose Logical; if TRUE, prints progress every 500 iterations.
#' @param seed Optional random seed for reproducibility.
#'
#' @return A list of posterior samples containing:
#' \describe{
#'   \item{\code{beta0}}{Vector of posterior samples for the intercept.}
#'   \item{\code{theta0}}{Matrix of posterior samples for modifying covariates in the intercept block (\eqn{\theta_0}).}
#'   \item{\code{beta}}{Matrix of posterior samples for main effects (\eqn{\beta}).}
#'   \item{\code{theta}}{Array of posterior samples for modifying effects of each predictor (\eqn{\theta_j}, dimensions p x q x n_samples).}
#'   \item{\code{sigma2}}{Vector of posterior samples for residual variance.}
#'   \item{\code{tau2}}{Vector of posterior samples for the global shrinkage parameter.}
#'   \item{\code{lambda2}}{Matrix of posterior samples for local shrinkage parameters of each predictor.}
#'   \item{\code{config}}{List of sampler configuration parameters (n_iter, burn_in, a0, b0, sigma0_sq, eps).}
#' }
#' @importFrom Rcpp sourceCpp
#' @examples
#' \dontrun{
#' n <- 100;
#' p <- 5;
#' q <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' Z <- matrix(rnorm(n * q), n, q)
#'
#' beta_true <- rep(0,p) ; beta_true[1:2] = c(-2,1)
#' theta_true <- matrix(rnorm(p * q), p, q)
#'
#' y <- X %*% beta_true + rowSums((Z %*% t(theta_true)) * X) + rnorm(n)
#'
#' y[sample(n, 10)] <- NA  # introduce missing values
#'
#' fit <- pliable_HS(y, X, Z, n_iter = 500, burn_in = 200)
#'
#' str(fit)
#'
#' be_HS_2 <- colMeans(fit$beta)
#' round( be_HS_2, 2 ) ; round( beta_true, 2 )
#' theta_HS_2 <- apply(fit$theta, c(1, 2), mean)
#' round( theta_HS_2, 2 ) ;
#' round( theta_true, 2 )
#' }
#'
#' @export
pliable_HS <- function(y, X, Z,
                                        n_iter = 2000, burn_in = 1000,
                                        a0 = 1e-2, b0 = 1e-2,
                                        sigma0_sq = 1.0,
                                        eps = 1e-6,
                                        verbose = FALSE,
                                        seed = NULL) {
  gibbs_pliable_lasso_missing_cpp(y_in = y,
                                  X = X, Z = Z,
                                  n_iter = n_iter,
                                  burn_in = burn_in,
                                  a0 = a0, b0 = b0,
                                  sigma0_sq = sigma0_sq,
                                  eps = eps,
                                  verbose = verbose,
                                  seed = seed)
}





#' Gibbs sampler for Logistic pliable lasso model with (Group) Horseshoe Prior for sparse interaction effects.
#'
#' This function implements a Gibbs sampler for logistic regression with
#' pliable lasso structure and a group horseshoe prior.
#' It is written in Rcpp and uses \pkg{RcppArmadillo} for efficiency.
#' The sampler relies on Polya-Gamma data augmentation and requires the
#' \pkg{BayesLogit} package for random draws from the Polya-Gamma distribution.
#'
#' @param y Numeric vector of binary outcomes of length \eqn{n}.
#' @param X Numeric matrix of predictors of dimension \eqn{n \times p}.
#' @param Z Numeric matrix of modifying variables of dimension \eqn{n \times q}.
#'   Must have the same number of rows as \code{X}.
#' @param n_iter Integer. Total number of Gibbs iterations. Default is 2000.
#' @param burn_in Integer. Number of burn-in iterations discarded from the
#'   beginning of the chain. Default is 1000.
#' @param sigma0_sq Prior variance for the intercept block (scalar). Default is 1.0.
#' @param eps Small ridge term added for numerical stability in matrix inversions.
#'   Default is 1e-6.
#' @param clamp_min Minimum allowed value for local/global shrinkage parameters
#'   (\eqn{\lambda^2}, \eqn{\tau^2}). Default is 1e-10.
#' @param clamp_max Maximum allowed value for local/global shrinkage parameters.
#'   Default is 1e10.
#' @param verbose Logical. If \code{TRUE}, prints progress every 500 iterations.
#' @author The Tien Mai, \email{the.tien.mai@@fhi.no}
#' @details
#' The model is
#' \deqn{ \text{logit}(P(y_i = 1)) = \beta_0 + Z_i \theta_0
#'       + \sum_{j=1}^p X_{ij} \big( \beta_j + Z_i \theta_j \big), }
#' where \eqn{\beta_j} are the main effects, \eqn{\theta_j} are modifier
#' effects associated with modifying variables \eqn{Z}, and the prior on each
#' group \eqn{(\beta_j, \theta_j)} is the group horseshoe:
#' \deqn{ (\beta_j, \theta_j) \sim \mathcal{N}\left(0, \tau^2 \lambda_j^2 I\right), }
#'
#' #' The regression coefficients (\eqn{\beta} for main effects and \eqn{\theta} for modifier interactions)
#' follow **hierarchical Horseshoe priors**, which allow for strong shrinkage of irrelevant
#' predictors while leaving relevant predictors relatively unshrunk:
#'
#' \deqn{\lambda_j^2 \sim \text{Inv-Gamma}(1/2, 1/ \nu_j), \quad \nu_j \sim \text{Inv-Gamma}(1/2, 1),}
#' \deqn{\tau^2 \sim \text{Inv-Gamma}(1/2, 1/\xi), \quad \xi \sim \text{Inv-Gamma}(1/2, 1).}
#'
#' Posterior inference is performed via Polya-Gamma data augmentation and
#' Gibbs sampling, following the algorithm implemented in C++ for speed.
#'
#' @return A list with the following elements:
#' \item{beta0}{Posterior samples of the intercept \eqn{\beta_0}.}
#' \item{theta0}{Posterior samples of the intercept modifier vector \eqn{\theta_0}.}
#' \item{beta}{Posterior samples of main effects \eqn{\beta_j}, dimension \eqn{(n_{\text{save}} \times p)}.}
#' \item{theta}{Posterior samples of modifier effects \eqn{\theta_j}, stored as an array with dimensions \eqn{(n_{\text{save}}, p, q)}.}
#' \item{tau2}{Posterior samples of the global scale parameter \eqn{\tau^2}.}
#' \item{lambda2}{Posterior samples of local scale parameters \eqn{\lambda_j^2}, dimension \eqn{(n_{\text{save}} \times p)}.}
#' \item{config}{List of configuration parameters used in the run.}
#'
#' @references
#' - Polson, N. G., Scott, J. G., & Windle, J. (2013).
#'   Bayesian inference for logistic models using Pólya–Gamma latent variables.
#'   *Journal of the American Statistical Association*, 108(504), 1339–1349.
#' - Mai. T.T. (2025). Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses. arXiv
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 100;
#' p <- 5;
#' q <- 2
#' X <- matrix(rnorm(n*p), n, p)
#' Z <- matrix(rnorm(n*q), n, q)
#' beta_true <- rnorm(p)
#' theta_true <- matrix(rnorm(p*q), p, q)
#'
#' eta <- 1 + X %*% beta_true + rowSums((X %*% theta_true) * Z)
#' prob <- 1/(1+exp(-eta))
#'
#' y <- rbinom(n, 1, prob)
#'
#' fit <- pliable_HS_logistic(y, X, Z, n_iter = 2000, burn_in = 1000)
#' str(fit)
#' }
#'
#' @export
#' @useDynLib hspliable
#' @importFrom Rcpp sourceCpp
pliable_HS_logistic <- function(y,
                                               X,
                                               Z,
                                               n_iter = 2000L,
                                               burn_in = 1000L,
                                               sigma0_sq = 1.0,
                                               eps = 1e-6,
                                               clamp_min = 1e-10,
                                               clamp_max = 1e10,
                                               verbose = TRUE) {
  gibbs_pliable_lasso_logistic2_rcpp(y,
                                     X,
                                     Z,
                                     n_iter = 2000L,
                                     burn_in = 1000L,
                                     sigma0_sq = 1.0,
                                     eps = 1e-6,
                                     clamp_min = 1e-10,
                                     clamp_max = 1e10,
                                     verbose = TRUE)
  }







#' Gibbs sampler for Poisson pliable lasso regression model with (Group) Horseshoe Prior
#'
#' This function implements a Gibbs sampler for a Poisson regression model with
#' pliable-lasso structure and a group horseshoe prior on the main + modifier groups.
#' It is written in Rcpp and uses \pkg{RcppArmadillo} for efficiency.
#'
#' @param y Numeric vector of count outcomes of length \eqn{n} (non-negative integers).
#' @param X Numeric matrix of predictors of dimension \eqn{n \times p}.
#' @param Z Numeric matrix of modifying variables of dimension \eqn{n \times q}.
#'   Must have the same number of rows as \code{X}.
#' @param n_iter Integer. Total number of Gibbs iterations. Default is 2000.
#' @param burn_in Integer. Number of burn-in iterations discarded from the
#'   beginning of the chain. Default is 1000.
#' @param sigma0_sq Prior variance for the intercept (scalar). Default is 1.0.
#' @param eps Small ridge term added for numerical stability in matrix inversions.
#'   Default is 1e-6.
#' @param clamp_min Minimum allowed value for local/global shrinkage parameters
#'   (\eqn{\lambda^2}, \eqn{\tau^2}). Default is 1e-10.
#' @param clamp_max Maximum allowed value for local/global shrinkage parameters.
#'   Default is 1e10.
#' @param verbose Logical. If \code{TRUE}, prints progress messages.
#'
#' @author The Tien Mai, \email{the.tien.mai@@fhi.no}
#'
#' @details
#' The Poisson pliable-lasso regression model uses a log link:
#' \deqn{ \log(\lambda_i) = \beta_0 + Z_i^\top \theta_0
#'       + \sum_{j=1}^p X_{ij} \big( \beta_j + Z_i^\top \theta_j \big), }
#' where \eqn{\lambda_i = E[y_i | X_i, Z_i]} and \eqn{\beta_j} are the main effects,
#' \eqn{\theta_j} are modifier effects (a q-vector per predictor), and
#' \eqn{\theta_0} is the intercept modifier vector.
#'
#' The groups \eqn{(\beta_j, \theta_j)} are assigned a group horseshoe prior:
#' \deqn{ (\beta_j, \theta_j) \sim \mathcal{N}\left(0, \tau^2 \lambda_j^2 I\right), }
#' together with the usual half-Cauchy/inverse-gamma parameterization for the
#' local scales \eqn{\lambda_j^2} and global scale \eqn{\tau^2}.
#'
#' Posterior inference is performed via Gibbs sampling implemented in C++ for speed.
#'
#' @return A list with elements:
#' \item{beta0}{Posterior samples of the intercept \eqn{\beta_0}.}
#' \item{theta0}{Posterior samples of the intercept modifier vector \eqn{\theta_0}.}
#' \item{beta}{Posterior samples of main effects \eqn{\beta_j} (saved iterations x p).}
#' \item{theta}{Posterior samples of modifier effects \eqn{\theta_j}, array with dimensions
#'   (saved iterations, p, q).}
#' \item{tau2}{Posterior samples of the global scale parameter \eqn{\tau^2}.}
#' \item{lambda2}{Posterior samples of local scale parameters \eqn{\lambda_j^2} (saved iterations x p).}
#' \item{config}{List of configuration parameters used in the run.}
#'
#' @references
#' - Mai. T.T. (2025). Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses. arXiv
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 100
#' p <- 10
#' q <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' Z <- matrix(rnorm(n * q), n, q)
#'
#' beta_true <- c(2, -2, 0, 2, rep(0, p - 4)) / 4
#' theta_true <- matrix(0, p, q)
#' theta_true[1:3, ] <- matrix(c(rep(1, q), rep(-2, q), c(1:q)), 3, q, byrow = TRUE) / 4
#' 
#' theta0_true <- rep(0.5, q)
#' beta0_true <- 2
#' # linear predictor and counts
#' eta <- beta0_true + Z %*% theta0_true +
#'         rowSums(sapply(1:p, function(j) X[, j] * (beta_true[j] + Z %*% theta_true[j, ])))
#' y <- rpois(n, lambda = exp(eta))
#'
#' fit <- pliable_HS_poisson(y, X, Z, n_iter = 5000L, burn_in = 1000L, verbose = TRUE)
#' colMeans(fit$beta)
#' apply(fit$theta, c(2, 3), mean)
#' }
#'
#' @export
#' @useDynLib hspliable
#' @importFrom Rcpp sourceCpp
pliable_HS_poisson <- function(y,
                               X,
                               Z,
                               n_iter = 2000L,
                               burn_in = 1000L,
                               sigma0_sq = 1.0,
                               eps = 1e-6,
                               clamp_min = 1e-10,
                               clamp_max = 1e10,
                               verbose = TRUE) {
  # Forward the actual function arguments to the compiled routine
  gibbs_pliable_lasso_poisson_rcpp(
     y,
    X = X,
    Z = Z,
    n_iter = as.integer(n_iter),
    burn_in = as.integer(burn_in),
    sigma0_sq = sigma0_sq,
    eps = eps,
    clamp_min = clamp_min,
    clamp_max = clamp_max,
    verbose = verbose
  )
}









#' Gibbs Sampler for the Bayesian Pliable Lasso with Sparse Interactions
#' for Gamma regression with possitive response (with missing data)
#'
#'
#' @param y Numeric POSITIVE response vector of length \eqn{n}. May include \code{NA}
#'   values, which will be imputed during sampling. Must satisfy \eqn{y_i > 0}
#'   for observed entries.
#' @param X Numeric \eqn{n \times p} predictor matrix.
#' @param Z Optional moderator matrix (\eqn{n \times q}). Interaction terms
#'   are created as \eqn{x_j * Z[,k]}.
#'
#' @param niter Total number of MCMC iterations.
#' @param burnin Number of burn-in iterations.
#' @param thin Thinning factor.
#'
#' @param b0_intercept Prior mean vector for the intercept block (confounder)
#'   \eqn{(\beta_0, \theta_0)}. Defaults to zero.
#' @param V0_intercept Prior covariance matrix for the intercept block (confounder).
#' @param a_tau, Shape/rate hyperparameters for the log-normal precision
#' @param b_tau Shape/rate hyperparameters for the log-normal precision
#'   \eqn{\tau_{\text{obs}}}.
#' @param sigma0_sq Prior variance for each component of the intercept block.
#'
#' @param tau2_init Initial global horseshoe scale squared.
#' @param beta_init Optional list containing initial values
#'   \code{beta0}, \code{theta0}, \code{beta}, \code{theta}.
#'
#' @param verbose Logical; print progress messages. Default is TRUE.
#' @param seed Optional random seed.
#'
#' @param eps Small diagonal ridge to stabilise inversions.
#' @param bound_k_min Lower bound on the Gamma shape estimator.
#'
#' @param save_imputed Logical; whether to store imputed missing \eqn{y}. Default is FALSE.
#' @param save_imputed_every Save imputed values every \eqn{m}-th saved iteration.
#'
#' @description
#' Implements a full Gibbs sampler for the Pliable Lasso model with
#' \strong{group horseshoe priors} on each interaction block.
#' The response is assumed to follow a positive distribution:
#'
#' \deqn{ Y_i \sim \mathcal{G}amma( \alpha , \alpha/ \eta_i). }
#'
#' The predictor structure is
#'
#' \deqn{
#' \eta_i = \beta_0 + Z_i^\top \theta_0 +
#'          \sum_{j=1}^p x_{ij} \left( \beta_j + Z_i^\top \theta_j \right).
#' }
#'
#' Each block \eqn{\gamma_j = (\beta_j, \theta_j)} has dimension
#' \eqn{d = 1 + q} and follows a group horseshoe prior:
#'
#' \deqn{
#' \gamma_j \mid \lambda_j^2, \tau^2 \sim
#'  \mathcal{N}_d\left(0, \tau^2 \lambda_j^2 I_d\right),
#' }
#'
#' with hierarchical horseshoe hyperpriors, half-Cauchy, on
#' the local scales \eqn{\lambda_j^2} and the global scale \eqn{\tau^2}.
#'
#' Missing \eqn{y_i} are imputed during sampling using the log-normal model.
#'
#' This sampler uses \code{MASS::mvrnorm()}, block-wise updates, and a
#' method-of-moments plug-in estimator for the implied Gamma shape parameter.
#'
#' @return A list with elements:
#' \describe{
#'   \item{beta0}{Posterior draws of \eqn{\beta_0}.}
#'   \item{theta0}{Posterior draws of \eqn{\theta_0}.}
#'   \item{beta}{Posterior draws of the \eqn{\beta_j}.}
#'   \item{theta}{Posterior draws of the interaction coefficients
#'                \eqn{\theta_{jk}}.}
#'   \item{tau_obs}{Posterior draws of the log-normal precision.}
#'   \item{tau2}{Posterior draws of the global horseshoe scale.}
#'   \item{k_hat}{Estimated Gamma shape parameter per iteration.}
#'   \item{lambda2, nu, xi}{Final values of horseshoe hyperparameters.}
#'   \item{config}{List containing run configuration.}
#'   \item{imputation}{List containing imputed missing values (if requested).}
#' }
#'
#'
#' @details
#' This sampler implements:
#'
#' * Block normal updates for each \eqn{\gamma_j}
#' * Full conditional update for the intercept block
#' * Horseshoe hierarchy using standard inverse-gamma augmentation
#' * Missing data imputation using the log-normal model
#' * Plug-in moment-based update for the Gamma shape parameter \eqn{k}
#'
#' The algorithm is stable for moderately high dimensions, as all updates
#' are conjugate and use Cholesky-based inverses.
#'
#'
#' @author The Tien Mai, \email{the.tien.mai@@fhi.no}
#' @references
#' - Tibshirani, R., & Friedman, J. (2020). A pliable lasso. *Journal of Computational and Graphical Statistics, 29*(1), 215-225.
#' - Mai. T.T. (2025). Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses. arXiv
#'
#' @examples
#' \dontrun{
#'
#'ntest <- 500
#'n <- 100
#'p <- 110
#'q <- 2
#'xx <- matrix(rnorm((n + ntest) * p), (n + ntest), p)
#'X <- xx[1:n, ]
#'xtest <- xx[-(1:n), ]
#'zz <- matrix(rnorm((n + ntest) * q), (n + ntest), q)
#'Z <- zz[1:n, ]
#'ztest <- zz[-(1:n), ]
#'beta_true <- c( .5,-2, 2, .5 , rep(0, p-4))
#'theta_true <- matrix(0, p, q)
#'theta_true[1:3, ] <- matrix( c(rep(1,q),
#'                               rep(-2,q),
#'                               c(1:q)) , 3, q, byrow = TRUE)
#'theta0_true = 0.5
#'beta0_true = 2
#'my_mu <- beta0_true + zz %*% rep(theta0_true, q) +
#'  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ] )))
#'mu_true <- exp(my_mu)
#'
#'k_true <- 2.0
#'yy <- rgamma(n, shape = k_true, scale = mu_true / k_true)
#'y <- yy[1:n]
#'
#'out_gibbs_HS <- gibbs_gamma_pliable_lognormal(y, X, Z ,
#'                                              niter = 8000, burnin = 1000, thin = 2,
#'                                              b0 = rep(0, 1 + q), V0 = diag(10, 1 + q),
#'                                              a_tau = 1, b_tau = 0.01,
#'                                              verbose = T, seed = 123)
#'round( colMeans(out_gibbs_HS$beta)[1:5], 3)
#'summary( out_gibbs_HS$k_hat )
#'sum( (  colMeans(out_gibbs_HS$beta) - beta_true )^2)
#'round( apply(out_gibbs_HS$theta, c(2,3), mean )[1:5,], 3)
#'theta_true[1:3,]
#'
#' # Fit with missing data
#'y_na = y
#'y_na[ sample(1:n, n*0.3) ] <- NA
#'library(tictoc); tic()
#'out_gibbs_HS <- gibbs_gamma_pliable_lognormal(y_na, X, Z ,
#'                                              niter = 5000, burnin = 1000, thin = 2,
#'                                              b0 = rep(0, 1 + q), V0 = diag(10, 1 + q),
#'                                              a_tau = 1, b_tau = 0.01,
#'                                              verbose = T, seed = 123)  ; toc()
#'round( colMeans(out_gibbs_HS$beta)[1:5], 3)
#'summary( out_gibbs_HS$k_hat )
#'sum( (  colMeans(out_gibbs_HS$beta) - beta_true )^2)
#'round( apply(out_gibbs_HS$theta, c(2,3), mean )[1:5,], 3)
#'theta_true[1:3,]
#'
#' }
#'
#' @export
pliable_HS_gamma_reg <- function(
    y, X, Z,
    niter = 5000, burnin = 1000, thin = 1,
    b0_intercept = NULL, V0_intercept = NULL,
    a_tau = 1.0, b_tau = 1e-2,
    sigma0_sq = 1.0,
    tau2_init = 1.0, prop = list(),
    beta_init = NULL,
    verbose = TRUE, seed = NULL,
    eps = 1e-8,
    bound_k_min = 1e-3,
    save_imputed = FALSE,
    save_imputed_every = 1
) {

  # -----------------------------
  # INITIAL CHECKS AND SETUP
  # -----------------------------
  if (!is.null(seed)) set.seed(seed)
  if (any(!is.na(y) & y <= 0))
    stop("Observed y must be positive for log-transform.")

  `%||%` <- function(a, b) if (!is.null(a)) a else b

  library(MASS)

  n <- length(y)
  X <- as.matrix(X)
  Z <- as.matrix(Z)
  p <- ncol(X)
  q <- if (is.matrix(Z)) ncol(Z) else 0
  d <- 1 + q

  # Prior for intercept block
  if (is.null(b0_intercept)) b0_intercept <- rep(0, d)
  if (is.null(V0_intercept)) V0_intercept <- diag(sigma0_sq, d)
  V0int_inv <- solve(V0_intercept)

  # ---------------------------------
  # INITIAL VALUES
  # ---------------------------------
  if (is.null(beta_init)) {
    beta0 <- 0
    theta0 <- if (q > 0) rep(0, q) else numeric(0)
    beta   <- rep(0, p)
    theta  <- if (q > 0) matrix(0, p, q) else matrix(0, p, 0)
  } else {
    beta0  <- beta_init$beta0  %||% 0
    theta0 <- beta_init$theta0 %||% (if (q>0) rep(0,q) else numeric(0))
    beta   <- beta_init$beta   %||% rep(0,p)
    theta  <- beta_init$theta  %||% (if (q>0) matrix(0,p,q) else matrix(0,p,0))
  }

  # Horseshoe
  lambda2 <- rep(1, p)
  nu <- rep(1, p)
  tau2 <- tau2_init
  xi <- 1

  rinv_gamma <- function(shape, rate) {
    if (shape <= 0 || rate <= 0) return(1e6)
    1 / rgamma(1, shape = shape, rate = rate)
  }

  # Storage
  nit_out <- floor((niter - burnin)/thin)
  if (nit_out < 1) stop("Not enough iterations after burnin.")

  out_beta0 <- numeric(nit_out)
  out_theta0 <- if (q>0) matrix(NA, nit_out, q) else NULL
  out_beta <- matrix(NA, nit_out, p)
  out_theta <- if (q>0) array(NA, c(nit_out, p, q)) else NULL
  out_tau_obs <- numeric(nit_out)
  out_tau2 <- numeric(nit_out)
  out_k_hat <- numeric(nit_out)

  # Missing data setup
  miss_idx <- which(is.na(y))
  n_miss <- length(miss_idx)
  y_obs_orig <- y
  y_work <- y

  if (save_imputed && n_miss > 0) {
    out_y_imputed <- matrix(NA, nit_out, n_miss)
    colnames(out_y_imputed) <- paste0("idx_", miss_idx)
  } else {
    out_y_imputed <- NULL
  }

  # Precompute design matrices
  W0 <- matrix(NA, n, d)
  W0[, 1] <- 1
  if (q > 0) W0[, 2:d] <- Z

  Wj_list <- vector("list", p)
  for (j in seq_len(p)) {
    xj <- X[, j]
    Wj <- matrix(NA, n, d)
    Wj[, 1] <- xj
    if (q>0)
      Wj[, 2:d] <- sweep(Z, 1, xj, "*")
    Wj_list[[j]] <- Wj
  }

  tau_obs <- 1

  # ---------------------------------
  # MCMC LOOP
  # ---------------------------------
  out_i <- 0

  for (iter in seq_len(niter)) {

    # ----- Linear predictor
    base_lin <- as.numeric(W0 %*% c(beta0, theta0))
    contrib <- matrix(0, n, p)
    eta <- base_lin
    for (j in seq_len(p)) {
      gj <- c(beta[j], theta[j, ])
      cj <- as.numeric(Wj_list[[j]] %*% gj)
      contrib[, j] <- cj
      eta <- eta + cj
    }

    # ----- Impute missing y
    if (n_miss > 0) {
      sigma2_curr <- 1 / tau_obs
      logy_miss <- rnorm(n_miss,
                         mean = eta[miss_idx],
                         sd = sqrt(sigma2_curr))
      y_work[miss_idx] <- pmax(exp(logy_miss), .Machine$double.xmin)
    }
    y_work[!is.na(y_obs_orig)] <- y_obs_orig[!is.na(y_obs_orig)]
    ylog <- log(y_work)

    # ----- Update intercept block
    resid0 <- ylog - rowSums(contrib)
    prec0 <- tau_obs * crossprod(W0) + V0int_inv + diag(eps, d)
    Sigma0 <- chol2inv(chol(prec0))
    mu0 <- Sigma0 %*% (tau_obs * crossprod(W0, resid0) + V0int_inv %*% b0_intercept)
    draw0 <- mu0 + chol(Sigma0) %*% rnorm(d)
    beta0 <- draw0[1]
    if (q>0) theta0 <- draw0[-1]

    # ----- Update each gamma_j block
    for (j in seq_len(p)) {
      Wj <- Wj_list[[j]]
      resid_j <- ylog - (base_lin + rowSums(contrib) - contrib[, j])
      prior_prec <- diag(1/(tau2*lambda2[j]), d)
      precj <- tau_obs * crossprod(Wj) + prior_prec + diag(eps, d)
      Sigma_j <- chol2inv(chol(precj))
      mu_j <- Sigma_j %*% (tau_obs * crossprod(Wj, resid_j))
      draw_j <- mu_j + chol(Sigma_j) %*% rnorm(d)
      beta[j] <- draw_j[1]
      if (q>0) theta[j, ] <- draw_j[-1]
      contrib[, j] <- as.numeric(Wj %*% draw_j)
    }

    eta <- base_lin + rowSums(contrib)

    # ----- Update tau_obs (precision)
    resid_all <- ylog - eta
    tau_obs <- rgamma(1,
                      a_tau + n/2,
                      b_tau + sum(resid_all^2)/2)

    # ----- Horseshoe local scales
    for (j in seq_len(p)) {
      gj <- c(beta[j], theta[j, ])
      g2 <- sum(gj^2)
      lambda2[j] <- rinv_gamma((d + 1)/2, 1/nu[j] + g2/(2*tau2))
      lambda2[j] <- max(lambda2[j], 1e-12)
      nu[j] <- rinv_gamma(0.5, 1 + 1/lambda2[j])
      nu[j] <- max(nu[j], 1e-12)
    }

    # ----- Horseshoe global scale
    total <- 0
    for (j in seq_len(p)) {
      gj <- c(beta[j], theta[j, ])
      total <- total + sum(gj^2)/lambda2[j]
    }
    tau2 <- rinv_gamma((p*d + 1)/2, 1/xi + total/2)
    tau2 <- max(tau2, 1e-12)
    xi <- rinv_gamma(0.5, 1 + 1/tau2)
    xi <- max(xi, 1e-12)

    # ----- Plug-in Gamma shape estimator
    sigma2 <- 1/tau_obs
    mu_hat <- exp(eta + sigma2/2)
    denom <- sum((y_work - mu_hat)^2)
    if (denom <= 0) k_hat <- 1e4 else
      k_hat <- max(sum(mu_hat^2) / denom, bound_k_min)

    # ----- Save
    if (iter > burnin && ((iter - burnin) %% thin == 0)) {
      out_i <- out_i + 1
      out_beta0[out_i] <- beta0
      if (q>0) out_theta0[out_i, ] <- theta0
      out_beta[out_i, ] <- beta
      if (q>0) out_theta[out_i, , ] <- theta
      out_tau_obs[out_i] <- tau_obs
      out_tau2[out_i] <- tau2
      out_k_hat[out_i] <- k_hat
      if (save_imputed && n_miss > 0 &&
          (out_i %% save_imputed_every == 0)) {
        out_y_imputed[out_i, ] <- y_work[miss_idx]
      }
    }

    if (verbose && iter %% max(1, niter%/%5) == 0) {
      cat(sprintf("iter %d/%d (saved %d): tau_obs=%.3g tau2=%.3g min(lambda2)=%.3g k=%.3g\n",
                  iter, niter, out_i,
                  tau_obs, tau2, min(lambda2), k_hat))
    }
  }

  # ---------------------------------
  # Return
  # ---------------------------------
  list(
    beta0 = out_beta0,
    theta0 = out_theta0,
    beta = out_beta,
    theta = out_theta,
    tau_obs = out_tau_obs,
    tau2 = out_tau2,
    lambda2 = lambda2,
    nu = nu,
    xi = xi,
    k_hat = out_k_hat,
    config = list(niter=niter, burnin=burnin, thin=thin, p=p, q=q, d=d),
    imputation = list(
      missing_idx = miss_idx,
      n_miss = n_miss,
      saved_imputations = out_y_imputed
    )
  )
}

