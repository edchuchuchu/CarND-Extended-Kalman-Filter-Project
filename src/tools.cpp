#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  if (estimations.size() == 0){
    cout << "Error: no estimations vector!" << endl;
    return rmse;
  }
//  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size()){
    cout << "Error: Vector size mismatch!" << endl;
    return rmse;
  }	

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i){
    VectorXd diff = (estimations[i] - ground_truth[i]).array().pow(2);
    rmse += diff;
  }

  //calculate the mean
  rmse /= estimations.size();
  //calculate the squared root
  rmse = rmse.array().sqrt();
  //return the result
  return rmse;  
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  Hj << 0, 0, 0, 0,
	0, 0, 0, 0,
	0, 0, 0, 0;	      
  //check division by zero
  if ((px == 0)&&(py == 0)){
    cout << "CalculateJacobian() - Error - Division by zero" << endl;
    return Hj;
  }
  //compute the Jacobian matrix
  Hj(0,0) = px / sqrt(pow(px, 2) + pow(py, 2));
  Hj(0,1) = py / sqrt(pow(px, 2) + pow(py, 2));
  Hj(1,0) = -py / (pow(px, 2) + pow(py, 2));
  Hj(1,1) = px / (pow(px, 2) + pow(py, 2));
  Hj(2,0) = (py*(vx * py - vy * px)) / pow((pow(px, 2) + pow(py, 2)), 1.5);
  Hj(2,1) = (px*(vy * px - vx * py)) / pow((pow(px, 2) + pow(py, 2)), 1.5);
  Hj(2,2) = px / sqrt(pow(px, 2) + pow(py, 2));
  Hj(2,3) = py / sqrt(pow(px, 2) + pow(py, 2));	

  return Hj;  
}
