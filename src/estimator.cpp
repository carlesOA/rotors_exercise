/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "estimator.h"
#include <tf_conversions/tf_eigen.h>
int n = 6;
int p = 3;
int m = 3;

Eigen::MatrixXd F(n,n); //(6,6)
Eigen::MatrixXd R(m,m);
Eigen::MatrixXd Q(n,n); //(6,6)
Eigen::MatrixXd P_hat(n,n); //(6,6)
Eigen::MatrixXd G(n,p); //(6,3)
Eigen::MatrixXd H(m,n); //(3,6)
Eigen::MatrixXd Ht(n,m); //(6,3)
Eigen::MatrixXd K(n,m); //(6,3)
Eigen::MatrixXd apriori_P;
Eigen::MatrixXd aposteriori_P;

Eigen::VectorXd x_hat(n);
Eigen::VectorXd u(p);
Eigen::VectorXd z(m);
Eigen::VectorXd apriori_x(n);
Eigen::VectorXd aposteriori_x(n);
Eigen::VectorXd apriori_z(m);

const double sigma_x = 0.001;
const float sigma_u = 0.5;
const double sigma_z = 0.001;

EstimatorNode::EstimatorNode() {


  ros::NodeHandle nh;

  pose_sub_ = nh.subscribe("/firefly/fake_gps/pose",  1, &EstimatorNode::PoseCallback, this);
  imu_sub_  = nh.subscribe("/firefly/imu",            1, &EstimatorNode::ImuCallback, this);


  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/firefly/pose", 1);

  timer_ = nh.createTimer(ros::Duration(0.1), &EstimatorNode::TimedCallback, this);

  nh.getParam("sigma_x", sigma_x);
  nh.getParam("sigma_z", sigma_z);
  nh.getParam("sigma_u", sigma_u);

  H << 1, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 0, 0,
       0, 0, 1, 0, 0, 0;

  Ht = H.transpose();

  R << sigma_z, 0, 0,
       0, sigma_z, 0,
       0, 0, sigma_z;

  apriori_P << sigma_x, 0, 0, 0, 0, 0,
       0, sigma_x, 0, 0, 0, 0,
       0, 0, sigma_x, 0, 0, 0,
       0, 0, 0, sigma_x, 0, 0,
       0, 0, 0, 0, sigma_x, 0,
       0, 0, 0, 0, 0, sigma_x;

  update(timer_);

}

EstimatorNode::~EstimatorNode() { }

void EstimatorNode::Publish()
{
  //publish your data
  //ROS_INFO("Publishing ...");

  pose_pub.publish(msgPose_);
}

void EstimatorNode::PoseCallback(
    const geometry_msgs::PoseStampedConstPtr& pose_msg) {

  ROS_INFO_ONCE("Estimator got first pose message.");
  msgPose_.header.stamp = pose_msg->header.stamp;
  msgPose_.header.seq = pose_msg->header.seq;
  msgPose_.header.frame_id =pose_msg->header.frame_id;

 //Correction
  K = P_hat * Ht * (H*P_hat*Ht + R).inverse();
  aposteriori_x = x_hat + K*(apriori_z - z);
  apriori_P = P_hat - K*H*P_hat;

}

void EstimatorNode::ImuCallback(
    const sensor_msgs::ImuConstPtr& imu_msg) {

  ROS_INFO_ONCE("Estimator got first IMU message.");

  msgPose_.header.stamp = imu_msg->header.stamp;
  msgPose_.header.seq = imu_msg->header.seq;
  msgPose_.header.frame_id =imu_msg->header.frame_id;

  apriori_x << 0,0,0,0,0,0;

  //PREDICTION
   apriori_x = F * x_hat + G;
   apriori_P = F * P_hat * F.transpose() +  Q;
   z = H * apriori_x;
   x_hat = apriori_x;
   P_hat = apriori_P;

}

void EstimatorNode::TimedCallback(
      const ros::TimerEvent& e){
   ROS_INFO_ONCE("Timer initiated.");
   Publish();
}

void EstimatorNode::update(double time){

    double dt = time;

    double dt_2 = dt*dt;
    double dt_3 = dt_2*dt;

    F << 1, 0, 0, dt, 0, 0,
         0, 1, 0, 0, dt, 0,
         0, 0, 1, 0, 0, dt,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    G << dt_2/2, 0, 0,
         0, dt_2/2, 0,
         0, 0, dt_2/2,
         dt, 0, 0,
         0, dt, 0,
         0, 0, dt;

    Q << dt_3/2, 0, 0, dt_2/2, 0, 0,
         0, dt_3/2, 0, 0, dt_2/2, 0,
         0, 0, dt_3/2, 0, 0, dt_2/2,
         dt_2/2, 0, 0, dt_2/2, 0, 0,
         0, dt_2/2, 0, 0, dt_2/2, 0,
         0, 0, dt_2/2, 0, 0, dt_2/2;

    Q = sigma_u*Q;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "estimator");

  EstimatorNode estimator_node;

  ros::spin();

  return 0;
}
