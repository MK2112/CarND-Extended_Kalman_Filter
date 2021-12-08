#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculating the RMSE.
   */

  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    cout << "Error: Sizes of estimations vector and ground truth vector not equal" << endl;
    return rmse;
  }

  // accumulating squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    rmse = rmse.array() + ((estimations[i] - ground_truth[i]).array() * (estimations[i] - ground_truth[i]).array());
  }

  // calculating the mean
  rmse = rmse/estimations.size();

  // calculating the squared root
  rmse = rmse.array().sqrt();

  // returning the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculating a Jacobian.
   */

  MatrixXd Hj(3,4);
  Hj << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // check division by zero
  if (pow(px,2) + pow(py, 2) == 0) {
    cout << "Error: Division by zero undefined in Jacobian calculation" << endl;
    return Hj;
  }
  // compute the Jacobian matrix
  Hj << (px/sqrt(pow(px,2) + pow(py, 2))), (py/sqrt(pow(px,2) + pow(py, 2))), 0, 0,
        -(py/(pow(px, 2) + pow(py, 2))), (px/(pow(px, 2) + pow(py, 2))), 0, 0,
        ((py*(vx*py-vy*px))/(pow(pow(px, 2)+pow(py, 2), 3/2))), ((px*(vy*px-vx*py))/(pow(pow(px, 2)+pow(py, 2), 3/2))), (px/sqrt(pow(px,2)+pow(py,2))), (py/sqrt(pow(px,2)+pow(py,2)));

  return Hj;
}
