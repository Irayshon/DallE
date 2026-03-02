#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "DallE/trajectory.h"
#include "robot_commander_cpp/quadruped_targets.hpp"

using namespace std::chrono_literals;

class QuadrupedIkCommander : public rclcpp::Node {
public:
  QuadrupedIkCommander()
  : Node("quadruped_ik_commander"), publish_count_(0) {
    pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/joint_trajectory_controller/joint_trajectory", 10);

    msg_.joint_names.assign(
      robot_commander_cpp::kQuadrupedJointNames.begin(),
      robot_commander_cpp::kQuadrupedJointNames.end());

    build_trajectory();
    timer_ = this->create_wall_timer(200ms, std::bind(&QuadrupedIkCommander::on_timer, this));
    RCLCPP_INFO(this->get_logger(), "quadruped IK commander ready");
  }

private:
  void build_trajectory() {
    Eigen::VectorXd start = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(robot_commander_cpp::kQuadrupedTargetPositions.size()));
    Eigen::VectorXd target = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(robot_commander_cpp::kQuadrupedTargetPositions.size()));
    for (Eigen::Index i = 0; i < target.size(); ++i) {
      target(i) = robot_commander_cpp::kQuadrupedTargetPositions[static_cast<std::size_t>(i)];
    }

    Eigen::MatrixXd trajectory = DallE::Trajectory::JointTrajectory(
      start,
      target,
      static_cast<double>(robot_commander_cpp::kQuadrupedTargetSeconds),
      2,
      5);

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.assign(target.size(), 0.0);
    for (Eigen::Index i = 0; i < target.size(); ++i) {
      point.positions[static_cast<std::size_t>(i)] = trajectory(trajectory.rows() - 1, i);
    }
    point.time_from_start.sec = robot_commander_cpp::kQuadrupedTargetSeconds;
    msg_.points.push_back(point);
  }

  void on_timer() {
    pub_->publish(msg_);
    publish_count_++;
    if (publish_count_ >= 10) {
      RCLCPP_INFO(this->get_logger(), "trajectory published 10 times, exiting");
      rclcpp::shutdown();
    }
  }

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  trajectory_msgs::msg::JointTrajectory msg_;
  int publish_count_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<QuadrupedIkCommander>());
  return 0;
}
