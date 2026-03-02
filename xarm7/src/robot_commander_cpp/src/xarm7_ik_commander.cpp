#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using namespace std::chrono_literals;

class Xarm7IkCommander : public rclcpp::Node {
public:
  Xarm7IkCommander()
  : Node("xarm7_ik_commander"), publish_count_(0) {
    pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/position_trajectory_controller/joint_trajectory", 10);

    msg_.joint_names = {
      "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {0.45, -1.2, 1.0, 0.5, 0.2, 1.2, 1.3};
    point.time_from_start.sec = 4;
    msg_.points.push_back(point);

    timer_ = this->create_wall_timer(200ms, std::bind(&Xarm7IkCommander::on_timer, this));
    RCLCPP_INFO(this->get_logger(), "xarm7 IK commander ready");
  }

private:
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
  rclcpp::spin(std::make_shared<Xarm7IkCommander>());
  return 0;
}
