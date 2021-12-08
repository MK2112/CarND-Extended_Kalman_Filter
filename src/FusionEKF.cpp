#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // initalizing matrices for state tranistion matrix F, state covariance matrix P and Q matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.Q_ = MatrixXd(4, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09,0,0,
              0,0.0009,0,
              0, 0, 0.09;

  // setting values for sensor matrix - laser
  H_laser_ << 1,0,0,0,
              0,1,0,0;

  // assigning initial values to F
  ekf_.F_ << 1,0,1,0,
             0,1,0,1,
             0,0,1,0,
             0,0,0,1;

  // assigning initial values to P, adjusted to initially high uncertainty
  ekf_.P_ << 1,0,0,0,
             0,1,0,0,
             0,0,999,0,
             0,0,0,999;   
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // converting radar from polar to cartesian coordinates and initializing state
      ekf_.x_(0) = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
      ekf_.x_(1) = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initializing laser state
      ekf_.x_(0) = measurement_pack.raw_measurements_[0];
      ekf_.x_(1) = measurement_pack.raw_measurements_[1];
    }

    // storing timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  float delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  float delta_t2 = pow(delta_t, 2);
  float delta_t3 = pow(delta_t, 3);
  float delta_t4 = pow(delta_t, 4);

  // saving timestamp for next prediction cycle
  previous_timestamp_ = measurement_pack.timestamp_; 

  // updating state transition matrix F according to new elapsed time, where time is measured in seconds
  ekf_.F_(0, 2) = delta_t;
  ekf_.F_(1, 3) = delta_t;

  float noise_ax = 9;
  float noise_ay = 9;

  ekf_.Q_ << delta_t4/4*noise_ax, 0, delta_t3/2*noise_ax, 0, 
             0, delta_t4/4*noise_ay, 0, delta_t3/2*noise_ay,
             delta_t3/2*noise_ax, 0, delta_t2*noise_ax, 0,
	           0, delta_t3/2*noise_ay, 0, delta_t2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // calculating Jacobian matrix over currently predicted state, modifying state transition matrix H to hold Jacobian Hj
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;

    // initializing measurement covariance matrix R, assigning radar related values
    ekf_.R_ = MatrixXd(3,3);
    ekf_.R_ = R_radar_; 
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    // making EKF use laser-specific state transition matrix H
    ekf_.H_ = H_laser_;

    // initializing the EKF's measurement covariance matrix R, assigning laser related (pre-defined) values
    ekf_.R_ = MatrixXd(2,2);
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}