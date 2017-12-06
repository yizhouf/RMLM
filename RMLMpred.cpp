//////////////////////////////
//
// Predictions and Inference with RML Models
// NOTE: all multivariate time series are stored with each timepoint being a column in  C++
//
/////////////////////////////

#include <Rcpp.h>
using namespace Rcpp;
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace::Eigen;

///////five assets, gspc based on msft, ge, jpm, xom
// transform time series into LMMR response
void ytrans(Ref<Eigen::VectorXd> y, Ref<Eigen::MatrixXd> ts) {
  int ii, jj;
  for(ii=0; ii<5; ii++){
    jj = ii*2;
    y(jj) = (ts(2,jj) - ts(1,jj))/exp(ts(1,jj+1));
    y(jj+1) = ts(2,jj+1) - ts(1,jj+1);
  }
  return;
}

// and to LMMR covariate
void xtrans(Ref<Eigen::VectorXd> x, Ref<Eigen::MatrixXd> ts){
  double tmp = exp(ts(1,1));
  int jj;
  x(0) = 1.0/tmp;
  for(int ii=1; ii<5; ii++){
    x(ii) = (ts(1,ii*2) - ts(0,ii*2))/tmp;
  }
  x(5) = ts(1,1);
  x(6) = 1.0;
  for(int ii=1; ii<5; ii++){
    jj = ii*3 + 4;
    x(jj) = 1.0/exp(ts(1,ii*2+1));
    x(jj+1) = ts(1, ii*2+1);
    x(jj+2) = 1.0;
  }
  return;
}

//[[Rcpp::export("InvT")]]
NumericMatrix InvT(Eigen::MatrixXd res, Eigen::MatrixXd ts0, int q, 
                   int nsamples, int lag){
  Eigen::MatrixXd ts(q, lag);
  int ii, jj, kk;
  
  for(ii=0; ii<nsamples; ii++){
    ts = ts0.block(0, ii*q, lag, q);
    for(jj=0; jj<5; jj++){
      kk = jj*2;
      res(ii,kk) = ts(1,kk) + res(ii,kk) * exp(ts(1,kk+1));
      res(ii,kk+1) = ts(1,kk+1) + res(ii,kk+1);
    }
  }
  
  return wrap(res);
}

//' XY model format from raw timeseries data 
//' @description 
//' XY model format from raw timeseries data
//' @params ts n*q matrix with q time series and n observations
//' @params emp apply empirical transformation or not
//' @params q the time series number
//' @params lag the lags of data involves in the X formation
//' @params N the length of time series
//' @params qq first qq time series will get empirical transformation
//' @params Pvec a vector of number of coefficient parameters in each equation
//' @details Base on Xtrans and Ytrans functions, this function transforms raw timeseries data
//' into the X and Y format needed in model.
//' @return The Y matrix, X List and empirical transformation information
//[[Rcpp::export("XYdata")]]
List ModelFormatXY(Eigen::MatrixXd ts, int q, int lag, int N, Eigen::VectorXi Pvec, 
                   Eigen::VectorXi cPvec) {
  int ii;
  
  Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> Y(N-lag, q);
  Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> X(N-lag, Pvec.sum());
  List XList(q);
  //List ksobj(qq);
  
  // compute transformations
  for(ii=0; ii<(N-lag); ii++) {
    ytrans(Y.row(ii), ts.block(ii, 0, lag+1, q));
    xtrans(X.row(ii), ts.block(ii, 0, lag, q));
  }
  
  return List::create(_["Y"] = wrap(Y), _["X"] = wrap(X));
}

//' simulation of time series.
//' for better efficiency, Beta is passed as a D x nReps matrix, and Sigma as a q x (q*nReps) matrix.
//' @description 
//' forward simulation based on the model structure
//' @params tsPred a matrix stores predictions on specific prediction periods
//' @params tsTmp temporary matrix
//' @params ts0 starting based data (depends on lags involved)
//' @params beta sigma the postior beta and sigma
//' @params iPred the specific prediction periods
//' @params q the time series number
//' @params lag the lags of data involves in the X formation
//' @params Pvec a vector of number of coefficient parameters in each equation
//' @details Base on Xtrans and Ytrans functions, this function transforms raw timeseries data
//' into the X and Y format needed in model.
//' @return ...
//[[Rcpp::export("rmlm_sim")]]
List multi_sim(Eigen::MatrixXd ts, Eigen::MatrixXd tsPred, Eigen::MatrixXd beta, 
               Eigen::MatrixXd sigma, int q, int d, int lag, int nsamples, 
               Eigen::VectorXi P, Eigen::VectorXi cP, bool first) {
  int ii, jj;
  
  Eigen::VectorXd xtemp(d);
  Eigen::VectorXd restemp(q);
  Eigen::VectorXd betat(d);
  Eigen::MatrixXd sigmat(q,q);
  
  if(first){
    for(ii=0; ii<nsamples; ii++) {
      sigma.block(q*ii,0,q,q) = sigma.block(q*ii,0,q,q).llt().matrixL();  
    }
  }
  
  for(ii=0; ii<nsamples; ii++) {
    betat = beta.row(ii);
    sigmat = sigma.block(q*ii,0,q,q);
    // simulate residual
    for(jj=0; jj<q; jj++) {
      restemp(jj) = norm_rand();
    }
    restemp = sigmat * restemp;
    // add mean
    xtrans(xtemp, ts.block(0,q*ii,lag,q));
    for(jj=0; jj<q; jj++) {
      restemp(jj) += 
        xtemp.segment(cP(jj), P(jj)).dot(betat.segment(cP(jj), P(jj)));
    }
    tsPred.row(ii) = restemp;
  }
  
  // invtrain(tsTmp.col(lag+ii), restemp, tsTmp.block(0,ii,q,lag));
  
  return List::create(_["tsPred"] = wrap(tsPred),
                      _["sigma"] = wrap(sigma));
}
