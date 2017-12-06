////////////////////////////////////////////
//
// Gibbs sampler for the ragged multivariable linear model (RMLM)
// Ethan Fang, version 2.0, Jan 2015
//
////////////////////////////////////////////

#include <Rcpp.h>
using namespace Rcpp;
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;

//' single Wishard distribution simulation
//' @description
//' One single random Wishard distribution generation
//' @params sigma the random generated matrix 
//' @params nu the degree of freedom in the Wishard distribution
//' @params q the underline matrix dimension
//' @params LLSigma the LLT object for Psi 
//' @details ...
//' @return Wishard matrix
void rWish(Ref<Eigen::MatrixXd> sigma, double nu, int q, LLT<Eigen::MatrixXd>& LLSigma)
{
  sigma = sigma.setZero();
  //standard wishard^{1/2} with nu, step 3
  for(int ii=0; ii<q; ii++){
    sigma(ii, ii) =  sqrt(Rf_rchisq(nu - ii));
    for(int jj=0; jj<ii; jj++)
      sigma(ii, jj) = norm_rand();
  }
  //wishard^{1/2} with nu and psi, step 4
  LLSigma.matrixU().solveInPlace(sigma);
  sigma = sigma*(sigma.transpose());
}


//' Sigma | Beta gibbs updating
//' @description Sigma | Beta gibbs updating algorithm
//' @params XXt XYt YYt: the temp matrices begin with sufficient statistics XX XY YY
//' @params P cP: the index vector for beta separated by equations
//' @params n: number of data
//' @params q: number of equations (assets)
//' @params d: number of beta parameters
//' @params beta: the vector updated in the previous step 
//' @params sigma: the matrix waiting to be updated 
//' @params psi nu: the prior parameters
//' @params LLSigma: inverse matrix needed temp matrices
//' @details ...
//' @return the updated sigma
void SigmaUpdate(Ref<Eigen::VectorXi> P, Ref<Eigen::VectorXi> cP, int n, int q, int d, 
                 Ref<Eigen::MatrixXd> psi, double nu,
                 Ref<Eigen::VectorXd> beta, Ref<Eigen::MatrixXd> sigma,
                 Ref<Eigen::MatrixXd> YYt, Ref<Eigen::MatrixXd> XYt, 
                 Ref<Eigen::MatrixXd> XXt, LLT<Eigen::MatrixXd>& LLSigma)
{
  int ii, jj;
  
  //compute sufficient statistics D,step 1
  for(ii=0; ii<d; ii++){
    XYt.row(ii) = -XYt.row(ii)*beta(ii);
    XXt.row(ii) = XXt.row(ii)*beta(ii);
    XXt.col(ii) = XXt.col(ii)*beta(ii);
  }
  for(ii=0; ii<q; ii++){
    for(jj=0; jj<P(ii); jj++)
      XYt.col(ii) += 0.5*XXt.col(cP(ii)+jj);
  }
  for(ii=0; ii<q; ii++){
    for(jj = 0; jj < P(ii); jj++){
      YYt.col(ii) += XYt.row(cP(ii)+jj);
      YYt.row(ii) += XYt.row(cP(ii)+jj);
    }
  }
  
  //compute psi for sigma and its cholesky, step 2
  LLSigma.compute(YYt+psi);
  
  //draw wishard distribution
  rWish(sigma, nu+n, q, LLSigma);
}

//' single Multi-Normal simulation
//' @description
//' random Multi-Normal distribution generation
//' @params beta the random generated vector 
//' @params mean the mean vector
//' @params d the normal distribution dimension
//' @params LLBeta the LLT object of var-cor matrix
//' @details ...
//' @return single multi-normal draw
void rNorm(Ref<Eigen::VectorXd> beta, Ref<Eigen::VectorXd> mean, 
           int d, LLT<Eigen::MatrixXd> LLBeta) 
{
  //step 3
  for(int ii=0; ii<d; ii++)
    beta(ii) = norm_rand();
  //step 4
  LLBeta.matrixU().solveInPlace(beta);
  beta += mean;
}


//' Beta | Sigma gibbs updating
//' @description Beta | Sigma gibbs updating algorithm
//' @params XY: the pre-calculated sufficient statistics
//' @params P cP: the index vector for beta separated by equations
//' @params n: number of data
//' @params q: number of equations (assets)
//' @params d: number of beta parameters
//' @params beta: the vector waiting to be updated 
//' @params sigma: the updated matrix in previous step 
//' @params Omegai lambda: the prior parameters
//' @params XXt, betahat: temp objects
//' @params LLBeta: the LLT object for temp matrix
//' @details ...
//' @return ...
void BetaUpdate(Ref<Eigen::MatrixXd> XY, Ref<Eigen::VectorXi> P, 
                Ref<Eigen::VectorXi> cP, int q, int d,
                Ref<Eigen::VectorXd> lambda, Ref<Eigen::MatrixXd> omegai,
                Ref<Eigen::MatrixXd> sigma, Ref<Eigen::VectorXd> beta,
                Ref<Eigen::MatrixXd> XXt, Ref<Eigen::VectorXd> betahat, 
                LLT<Eigen::MatrixXd> LLBeta) 
{
  int ii, jj, kk;
  
  //sufficient statistics XX*sigma into XXt, step 1
  for(ii = 0; ii < q; ii++)
    for(jj = 0; jj < q; jj++)
      XXt.block(cP(ii), cP(jj), P(ii), P(jj)) *= sigma(ii,jj);
  
  //sufficient statistics Sigma^{-1}betahat into betahat
  betahat = betahat.setZero();
  for(jj = 0; jj < q; jj++)
    for(ii = 0; ii < q; ii++)
      for(kk = 0; kk < P(ii); kk++)
        betahat(cP(ii)+kk) += XY(cP(ii)+kk, jj)*sigma(ii, jj);
  //lambda is already Omega^{-1}lambda
  betahat += lambda;
  //compute normal mean, step 2
  LLBeta.compute(XXt + omegai);
  LLBeta.matrixL().solveInPlace(betahat);
  LLBeta.matrixU().solveInPlace(betahat);
  
  rNorm(beta, betahat, d, LLBeta);
}

//' Gibbs sampler for RMLM
//' @description The main function for Gibbs sampler for RMLM
//' @params Y X the input variates
//' @params P cP: the index vector for beta separated by equations
//' @params psi nu lambda omega: parameters for the prior
//' @params beta0 the starting value for the algorithm (start with sigma|beta) 
//' @params n: number of data
//' @params q: number of equations (assets)
//' @params d: number of beta parameters
//' @params nSamples posterior sample size
//' @params nBrun burning sample size
//' @params si return inverse sigma if si = TRUE
//' @details ...
//' @return List of posterior Beta and Sigma (Inverse Sigma)
//[[Rcpp::export("lmmrGibbs")]]
List lmmrGibbs(Eigen::MatrixXd Y, Eigen::MatrixXd X, Eigen::VectorXi P, 
               Eigen::VectorXi cP, Eigen::MatrixXd psi, double nu, 
               Eigen::VectorXd lambda, Eigen::MatrixXd omega, 
               Eigen::VectorXd beta0, int q, int n, int d, int nSamples, int nBurn,
               bool si) 
{
  int ii;
  MatrixXd omegai = omega;
  if(!omegai.isZero(0)) {
    omegai = omega.inverse();
  }
  lambda = omegai*lambda;
  
  MatrixXd sigma(q, q);
  VectorXd beta = beta0;
  
  MatrixXd XX = (X.transpose())*X;
  MatrixXd XY = (X.transpose())*Y;
  MatrixXd YY = (Y.transpose())*Y;
  
  MatrixXd YYt(q,q);
  MatrixXd XYt(d,q);
  MatrixXd XXt(d,d);
  VectorXd betahat = beta;
  
  MatrixXd PostBeta(nSamples, d);
  MatrixXd PostSigma(nSamples*q, q);
  MatrixXd PostSigmain(nSamples*q, q);
  
  LLT<MatrixXd> LLSigma(YYt);
  LLT<MatrixXd> LLBeta(XXt);
  
  for(ii = -nBurn; ii < nSamples; ii++){
    YYt = YY;
    XYt = XY;
    XXt = XX;
    SigmaUpdate(P, cP, n, q, d, psi, nu, beta, sigma, YYt, XYt, XXt, LLSigma);
    XXt = XX;
    BetaUpdate(XY, P, cP, q, d, lambda, omegai, sigma, beta, XXt, betahat, LLBeta);
    if(ii >= 0) {
      if(si)
        PostSigmain.block(ii*q, 0, q, q) = sigma;
      PostSigma.block(ii*q, 0, q, q) = sigma.inverse();
      PostBeta.row(ii) = beta;
    }
  }
  if(si)
    return List::create(_["Beta"] = wrap(PostBeta), _["Sigma"] = wrap(PostSigma), 
                        _["Sigmain"] = wrap(PostSigmain));
  else
    return List::create(_["Beta"] = wrap(PostBeta), _["Sigma"] = wrap(PostSigma));
}
