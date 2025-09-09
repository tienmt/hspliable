// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_USE_CURRENT
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// ---- small helpers ---------------------------------------------------------
// draw from inverse-gamma via 1/rgamma(shape, rate)
inline double rinv_gamma_1(double shape, double rate) {
  double g = R::rgamma(shape, 1.0 / rate); // R::rgamma uses 'scale', so pass scale = 1/rate
  // guard against pathological 0
  if (g <= 0.0 || !R_finite(g)) g = std::numeric_limits<double>::min();
  return 1.0 / g;
}

// draw from N(m, V) using Cholesky; V must be SPD (we add tiny ridge upstream)
inline arma::vec rmvnorm_1(const arma::vec& m, const arma::mat& V) {
  arma::mat L = arma::chol(V, "lower");    // if fails, caller should have stabilized V already
  arma::vec z = arma::randn<arma::vec>(m.n_elem);
  return m + L * z;
}

// [[Rcpp::export]]
Rcpp::List gibbs_pliable_lasso_missing_cpp(const arma::vec& y_in,
                                           const arma::mat& X,
                                           const arma::mat& Z,
                                           int n_iter   = 2000,
                                           int burn_in  = 1000,
                                           double a0 = 1e-2, double b0 = 1e-2,   // IG prior for sigma^2
                                           double sigma0_sq = 1.0,               // prior var for (beta0, theta0)
                                           double eps = 1e-6,                    // ridge for stability
                                           bool verbose = false,
                                           Rcpp::Nullable<int> seed = R_NilValue) {
  RNGScope scope; // ensure R RNG is used for R::rnorm/rgamma etc.

  if (seed.isNotNull()) {
    int s = Rcpp::as<int>(seed);
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed = base_env["set.seed"];
    set_seed(s);
  }

  const int n = X.n_rows;
  const int p = X.n_cols;
  const int q = Z.n_cols;

  if ((int)y_in.n_elem != n || (int)Z.n_rows != n) {
    stop("Dimension mismatch: length(y) must equal nrow(X) and nrow(Z).");
  }

  // Work copies
  arma::vec y = y_in;

  // observed / missing indices (fixed over the chain)
  arma::uvec obs_idx = arma::find_finite(y);
  arma::uvec mis_idx = arma::find_nonfinite(y);

  // ----- parameters -----
  const int d = 1 + q;

  double beta0 = 0.0;
  arma::vec theta0(q, fill::zeros);

  arma::vec  beta(p, fill::zeros);
  arma::mat  theta(p, q, fill::zeros);

  arma::vec lambda2(p, fill::ones);
  arma::vec nu(p, fill::ones);
  double tau2 = 1.0;
  double xi   = 1.0;

  // initialize missing y (simple: zeros)
  if (mis_idx.n_elem > 0) {
    y.elem(mis_idx).zeros();
  }

  // initialize sigma2 from observed variance if possible
  double sigma2;
  if (obs_idx.n_elem > 1) {
    arma::vec yo = y.elem(obs_idx);
    double m = arma::mean(yo);
    double v = arma::mean(arma::square(yo - m));
    sigma2 = (R_finite(v) && v > 0.0) ? v : 1.0;
  } else {
    sigma2 = 1.0;
  }

  // storage
  const int keep = n_iter - burn_in;
  if (keep <= 0) stop("burn_in must be < n_iter.");
  arma::vec beta0_store(keep, fill::zeros);
  arma::mat theta0_store(keep, q, fill::zeros);
  arma::mat beta_store(keep, p, fill::zeros);
  arma::cube theta_store(p, q, keep, fill::zeros); // (p x q x keep), same as your earlier C++ layout
  arma::vec sigma2_store(keep, fill::zeros);
  arma::vec tau2_store(keep, fill::zeros);
  arma::mat lambda2_store(keep, p, fill::zeros);

  // precompute ones and (1, Z) block design per iter (we rebuild W0 each iter as needed)
  arma::vec one_n(n, fill::ones);

  // ----- main Gibbs loop -----
  for (int iter = 1; iter <= n_iter; ++iter) {
    // 1) Data augmentation: y_mis | rest ~ N(mu_mis, sigma2)
    if (mis_idx.n_elem > 0) {
      // fitted = beta0 + Z*theta0 + X*beta + rowSums( (Z * t(theta)) % X )
      arma::mat Z_theta_t = Z * theta.t();             // n x p
      arma::vec interaction = arma::sum(Z_theta_t % X, 1); // rowSums -> n x 1
      arma::vec fitted = beta0 + (Z * theta0) + (X * beta) + interaction;
      arma::vec mu_mis = fitted.elem(mis_idx);

      // elementwise: y_mis = mu_mis + sqrt(sigma2) * N(0,1)
      arma::vec noise = std::sqrt(sigma2) * arma::randn<arma::vec>(mis_idx.n_elem);
      y.elem(mis_idx) = mu_mis + noise;
    }

    // Precompute current fitted pieces for speed
    arma::mat Z_theta_t = Z * theta.t();                 // n x p
    arma::vec eta_all   = (X * beta) + arma::sum(Z_theta_t % X, 1); // n
    arma::vec base_lin  = beta0 + (Z * theta0);          // n
    arma::vec fitted    = base_lin + eta_all;
    arma::vec resid_all = y - fitted;

    // 2) Update intercept block (beta0, theta0)
    // W0 = [1, Z]; P0 = (W0'W0)/sigma2 + I/sigma0_sq (+ ridge)
    arma::mat W0(n, 1 + q);
    W0.col(0) = one_n;
    W0.cols(1, q) = Z;

    arma::mat P0 = (W0.t() * W0) / sigma2 + arma::eye(1 + q, 1 + q) / sigma0_sq;
    P0.diag() += eps;
    arma::mat V0 = arma::inv_sympd(P0);  // SPD by construction + ridge
    arma::vec m0 = V0 * (W0.t() * (y - eta_all) / sigma2);
    arma::vec par0 = rmvnorm_1(m0, V0);
    beta0  = par0(0);
    theta0 = par0.subvec(1, q);

    // refresh base / fitted / residuals
    base_lin  = beta0 + (Z * theta0);
    fitted    = base_lin + eta_all;
    resid_all = y - fitted;

    // 3) For each j, update (beta_j, theta_j) jointly
    for (int j = 0; j < p; ++j) {
      arma::vec xj = X.col(j);                         // n
      arma::mat Zj = Z.each_col() % xj;               // n x q, diag(xj) * Z

      // current contribution of group j
      arma::vec contrib_j = xj * beta(j) + Zj * theta.row(j).t();

      // partial residual excluding j
      arma::vec r_minus_j = y - base_lin - (eta_all - contrib_j);

      // Wj = [xj, Zj] (n x d)
      arma::mat Wj(n, d);
      Wj.col(0) = xj;
      Wj.cols(1, q) = Zj;

      // Posterior for gamma_j = (beta_j, theta_j): N(mj, Vj)
      arma::mat Pj = (Wj.t() * Wj) / sigma2 + arma::eye(d, d) / (tau2 * lambda2(j));
      Pj.diag() += eps;
      arma::mat Vj = arma::inv_sympd(Pj);
      arma::vec mj = Vj * (Wj.t() * r_minus_j / sigma2);
      arma::vec gamma_j = rmvnorm_1(mj, Vj);

      // assign back
      beta(j)        = gamma_j(0);
      theta.row(j)   = gamma_j.subvec(1, q).t();

      // update cached totals efficiently
      arma::vec new_contrib_j = xj * beta(j) + Zj * theta.row(j).t();
      eta_all += (new_contrib_j - contrib_j);
      fitted   = base_lin + eta_all;
      resid_all = y - fitted;
    }

    // 4) Update local scales lambda2_j and auxiliaries nu_j
    for (int j = 0; j < p; ++j) {
      double g2 = beta(j) * beta(j) + arma::dot(theta.row(j), theta.row(j));
      double shape_lam = (d + 1.0) / 2.0;
      double rate_lam  = (1.0 / nu(j)) + (g2 / (2.0 * tau2));
      lambda2(j) = rinv_gamma_1(shape_lam, rate_lam);
      // nu_j ~ IG(1/2, 1 + 1/lambda2_j)
      nu(j) = rinv_gamma_1(0.5, 1.0 + 1.0 / lambda2(j));
    }

    // 5) Update global scale tau2 and auxiliary xi
    double sum_g2_over_lam = 0.0;
    for (int j = 0; j < p; ++j) {
      double g2 = beta(j) * beta(j) + arma::dot(theta.row(j), theta.row(j));
      sum_g2_over_lam += g2 / lambda2(j);
    }
    double shape_tau = (p * d + 1.0) / 2.0;
    double rate_tau  = (1.0 / xi) + 0.5 * sum_g2_over_lam;
    tau2 = rinv_gamma_1(shape_tau, rate_tau);
    xi   = rinv_gamma_1(0.5, 1.0 + 1.0 / tau2);

    // 6) Update sigma2 | residuals from completed y
    double shape_sig = a0 + n / 2.0;
    double rate_sig  = b0 + 0.5 * arma::dot(resid_all, resid_all);
    sigma2 = rinv_gamma_1(shape_sig, rate_sig);

    // 7) store
    if (iter > burn_in) {
      int k = iter - burn_in - 1;
      beta0_store(k) = beta0;
      theta0_store.row(k) = theta0.t();
      beta_store.row(k)   = beta.t();
      theta_store.slice(k) = theta;
      sigma2_store(k) = sigma2;
      tau2_store(k)   = tau2;
      lambda2_store.row(k) = lambda2.t();
    }

    if (verbose && (iter % 500 == 0)) {
      Rcpp::Rcout << "iter " << iter
                  << ": sigma2=" << std::setprecision(4) << sigma2
                  << ", tau2="   << std::setprecision(4) << tau2
                  << std::endl;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("beta0")   = beta0_store,
    Rcpp::Named("theta0")  = theta0_store,
    Rcpp::Named("beta")    = beta_store,
    Rcpp::Named("theta")   = theta_store,
    Rcpp::Named("sigma2")  = sigma2_store,
    Rcpp::Named("tau2")    = tau2_store,
    Rcpp::Named("lambda2") = lambda2_store,
    Rcpp::Named("config")  = Rcpp::List::create(
      Rcpp::Named("n_iter") = n_iter,
      Rcpp::Named("burn_in") = burn_in,
      Rcpp::Named("a0") = a0,
      Rcpp::Named("b0") = b0,
      Rcpp::Named("sigma0_sq") = sigma0_sq,
      Rcpp::Named("eps") = eps
    )
  );
}
