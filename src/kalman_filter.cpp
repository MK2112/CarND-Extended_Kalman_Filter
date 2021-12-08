#include "kalman_filter.h"
#include <iostream>

# define PI 3.14159265359

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // deriving new estimate from old estimate and state transition matrix
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  // updating covariance matrix
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * S.inverse();

  // updating estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // updating covariance matrix
  P_ = (I - K * H_) * P_;
}

VectorXd hFunction(const VectorXd x) {
  // realizing the h function
  VectorXd z_pred(3);
  z_pred(0) = sqrt(pow(x(0), 2) + pow(x(1), 2));
  z_pred(1) = atan2(x(1), x(0));
  z_pred(2) = ((x(0)*x(2)+x(1)*x(3))/sqrt(pow(x(0), 2) + pow(x(1), 2)));
  
  if (z_pred(0) < 0.0001) {
    cout << "Warning: Prediction value very small, bumping to 0.0004 for better zero division avoidance" << endl;
    z_pred(0) = 0.0004;
  }
  return z_pred;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  // performing measurement update
  VectorXd y = z - hFunction(x_);

  if (y(1) > PI) {
    y(1) -= 2*PI;
  } else if (y(1) < -PI) {
    y(1) += 2*PI;
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // building new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}