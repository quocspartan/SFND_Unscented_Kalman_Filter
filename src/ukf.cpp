#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = Eigen::VectorXd::Zero(5);

  // initial covariance matrix
  P_ = Eigen::MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  is_initialized_ = false; 
  // state dimension
  n_x_ = 5; 
  // augmented state dimension
  n_aug_ = n_x_ + 2; 
  // predicted sigma points matrix
  Xsig_pred_ = Eigen::MatrixXd::Zero(n_x_, 2*n_aug_ + 1);
  // weight of sigma points
  weights_ = Eigen::VectorXd::Zero(2*n_aug_ + 1); 
  // sigma point spreading param
  lambda_ = 3.0 - n_x_; 

  // update P_ to a identity matrix
  for (auto i = 0; i < n_x_; i++)
    P_(i,i) = 1.0; 

  // update weights
  weights_(0) = lambda_/ (lambda_ + n_aug_);
  for (auto i = 0; i < 2*n_aug_ + 1; i++)
    weights_(i) = 0.5 / (lambda_ + n_aug_);    
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (!is_initialized_) {
    //simulation start, setting the initial state values
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      //5x1 state vector
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],0,0,0;
      //5x5 covariance matric
      P_ << std_laspx_*std_laspx_,0,0,0,0,
            0, std_laspy_*std_laspy_,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1;

    } else {
      //radar measurements
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rhodot = meas_package.raw_measurements_[2];
      double px = rho*cos(phi);
      double py = rho*sin(phi);
      x_ << px, py, rhodot, phi, rhodot;
      P_ << std_radr_*std_radr_,0,0,0,0,
            0, std_radr_*std_radr_,0,0,0,
            0,0, std_radrd_*std_radrd_,0,0,
            0,0,0, std_radphi_*std_radphi_,0,
            0,0,0,0, std_radphi_*std_radphi_;
    }
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
  } else {
    //compute the elapsed time
    float delta_t = (meas_package.timestamp_ - time_us_) / 1e6;
    time_us_ = meas_package.timestamp_;

    Prediction(delta_t);
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {
      UpdateLidar(meas_package);
    } else {
      UpdateRadar(meas_package);
    }
  }

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  VectorXd x_aug = Eigen::VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_; 
  std::cout << "DEBUG: x_aug: " << x_aug << std::endl; 

  MatrixXd P_aug = Eigen::MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;
  std::cout << "DEBUG: P_aug: " << P_aug << std::endl; 

  // create the square root matrix used for generating sigmas
  MatrixXd A = P_aug.llt().matrixL();
  //create the aug sigmas
  MatrixXd X_sig_aug = Eigen::MatrixXd::Zero(n_aug_, 2*n_aug_+1);
  X_sig_aug.col(0) = x_aug;
  for(auto i = 0; i < n_aug_; i++){
    X_sig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*A.col(i);
    X_sig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*A.col(i);
  }
  std::cout << "DEBUG: X_sig_aug: " << X_sig_aug << std::endl; 
  
  // sigma points prediction
  Xsig_pred_ = Eigen::MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  for (auto i = 0; i< 2*n_aug_+1; ++i) {    
    double p_x = X_sig_aug(0,i);
    double p_y = X_sig_aug(1,i);
    double v = X_sig_aug(2,i);
    double yaw = X_sig_aug(3,i);
    double yawd = X_sig_aug(4,i);
    double nu_a = X_sig_aug(5,i);
    double nu_yawdd = X_sig_aug(6,i);    
    double px_p, py_p;

    // check if yawd is zero
    if (fabs(yawd) > __FLT_EPSILON__) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // update predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }  
  std::cout << "DEBUG: Xsig_pred_: " << Xsig_pred_ << std::endl; 
  
  // compute predicted mean
  x_.fill(0.0);
  for (auto i = 0; i < 2 * n_aug_ + 1; i++) { 
    x_ += weights_(i) * Xsig_pred_.col(i);
  }
  while (x_(3) >  M_PI) x_(3) -= 2*M_PI;
  while (x_(3) < -M_PI) x_(3) += 2*M_PI;
  
  // compute predicted covariance
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) { 
    // difference between each predicted sigma point and their mean
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization to -pi to pi
    while (x_diff(3) > M_PI) x_diff(3) -= 2*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
  std::cout << "DEBUG: P_: " << P_ << std::endl; 
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}