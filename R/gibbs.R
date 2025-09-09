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
#' n <- 100; p <- 5; q <- 2
#' X <- matrix(rnorm(n * p), n, p)
#' Z <- matrix(rnorm(n * q), n, q)
#' beta_true <- rnorm(p)
#' theta_true <- matrix(rnorm(p * q), p, q)
#' y <- X %*% beta_true + rowSums((Z %*% t(theta_true)) * X)
#'
#' y[sample(n, 10)] <- NA  # introduce missing values
#'
#' fit <- pliable_HS(y, X, Z, n_iter = 500, burn_in = 200)
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





#' Gibbs sampler for Logistic pliable Horseshoe with (Group) Horseshoe Prior for sparse interaction effects.
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
#' n <- 100; p <- 5; q <- 2
#' X <- matrix(rnorm(n*p), n, p)
#' Z <- matrix(rnorm(n*q), n, q)
#' beta_true <- rnorm(p)
#' theta_true <- matrix(rnorm(p*q), p, q)
#' eta <- 1 + X %*% beta_true + rowSums((X %*% theta_true) * Z)
#' prob <- 1/(1+exp(-eta))
#' y <- rbinom(n, 1, prob)
#'
#' fit <- pliable_HS_logistic(y, X, Z, n_iter = 2000, burn_in = 1000)
#' str(fit)
#' }
#'
#' @export
#'
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
