"""TB6-R5 hardware teleoperation entry point.

Usage:
    python scripts/hardware/teleop_tb6r5_hardware.py
    python scripts/hardware/teleop_tb6r5_hardware.py --robot-ip 192.168.1.100
    python scripts/hardware/teleop_tb6r5_hardware.py --robot-ip none --visualize-placo
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode jog_any_c
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode jog_any_c --no-jog-any-c-position-only
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode jog_any_c --jog-any-c-orientation-only
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode jog_any_c --jog-any-c-interrupt on
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode placo_ik --zone-ratio 0.05
    python scripts/hardware/teleop_tb6r5_hardware.py --scale-factor 2.0 --cartesian-max-step-pos-m 0.05
    python scripts/hardware/teleop_tb6r5_hardware.py --teleop-mode jog_any_c --cartesian-vel 1.0 --cartesian-acc 1.0 --cartesian-dec 1.0
    python scripts/hardware/teleop_tb6r5_hardware.py --jog-any-c-preview
    python scripts/visualization/vis_jog_any_c_robottarget.py
    python scripts/hardware/teleop_tb6r5_hardware.py --enable-log-data --log-dir logs/tb6r5
    python scripts/hardware/teleop_tb6r5_hardware.py --enable-log-data --enable-camera
"""

import tyro

from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS,
    DEFAULT_JOG_ANY_JOINT_ACC,
    DEFAULT_JOG_ANY_JOINT_DEC,
    DEFAULT_JOG_ANY_JOINT_VEL,
    DEFAULT_TELEOP_MODE,
    TeleopMode,
)
from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    TB6R5TeleopController,
    DEFAULT_URDF_PATH,
    DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    DEFAULT_ROBOT_IP,
    DEFAULT_RPC_PORT,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_CARTESIAN_MAX_STEP_POS_M,
    DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD,
    DEFAULT_JOG_ANY_C_POSITION_ONLY,
    DEFAULT_JOG_ANY_C_ORIENTATION_ONLY,
    DEFAULT_JOG_ANY_C_INTERRUPT,
    DEFAULT_ZONE_RATIO,
    DEFAULT_LOG_JOINT_COUNT,
    DEFAULT_GRIPPER_TRIGGER_NAME,
    DEFAULT_GRIPPER_TRIGGER_THRESHOLD,
    DEFAULT_GRIPPER_OPEN_CMD,
    DEFAULT_GRIPPER_CLOSED_CMD,
    DEFAULT_GRIPPER_OBSERVATION,
    DEFAULT_REALSENSE_SERIAL_DICT,
    JogAnyCInterruptMode,
)


def main(
    robot_urdf_path: str = DEFAULT_URDF_PATH,
    manipulator_config: dict = DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    robot_ip: str = DEFAULT_ROBOT_IP,
    rpc_port: int = DEFAULT_RPC_PORT,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    cartesian_max_step_pos_m: float = DEFAULT_CARTESIAN_MAX_STEP_POS_M,
    cartesian_max_step_rot_rad: float = DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD,
    jog_any_c_position_only: bool = DEFAULT_JOG_ANY_C_POSITION_ONLY,
    jog_any_c_orientation_only: bool = DEFAULT_JOG_ANY_C_ORIENTATION_ONLY,
    jog_any_c_interrupt: JogAnyCInterruptMode = DEFAULT_JOG_ANY_C_INTERRUPT,
    zone_ratio: float = DEFAULT_ZONE_RATIO,
    jog_any_c_async_timeout_ms: int = DEFAULT_JOG_ANY_C_ASYNC_TIMEOUT_MS,
    cartesian_vel: float | None = None,
    cartesian_acc: float | None = None,
    cartesian_dec: float | None = None,
    joint_vel: float = DEFAULT_JOG_ANY_JOINT_VEL,
    joint_acc: float = DEFAULT_JOG_ANY_JOINT_ACC,
    joint_dec: float = DEFAULT_JOG_ANY_JOINT_DEC,
    visualize_placo: bool = False,
    control_rate_hz: int = 50,
    enable_log_data: bool = True,
    log_dir: str = "logs/tb6r5",
    log_freq: float = 50,
    enable_camera: bool = True,
    camera_serial_dict: dict[str, str] = DEFAULT_REALSENSE_SERIAL_DICT,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fps: int = 30,
    enable_camera_depth: bool = True,
    enable_camera_compression: bool = True,
    camera_jpg_quality: int = 85,
    log_joint_count: int = DEFAULT_LOG_JOINT_COUNT,
    gripper_trigger_name: str = DEFAULT_GRIPPER_TRIGGER_NAME,
    gripper_trigger_threshold: float = DEFAULT_GRIPPER_TRIGGER_THRESHOLD,
    gripper_open_cmd: float = DEFAULT_GRIPPER_OPEN_CMD,
    gripper_closed_cmd: float = DEFAULT_GRIPPER_CLOSED_CMD,
    gripper_observation_default: float = DEFAULT_GRIPPER_OBSERVATION,
    teleop_mode: TeleopMode = DEFAULT_TELEOP_MODE,
    jog_any_c_preview: bool = False,
):
    if jog_any_c_preview:
        robot_ip = "none"
        teleop_mode = "jog_any_c"
        visualize_placo = True
        enable_log_data = False
        enable_camera = False

    controller = TB6R5TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=manipulator_config,
        robot_ip=robot_ip,
        rpc_port=rpc_port,
        teleop_mode=teleop_mode,
        scale_factor=scale_factor,
        cartesian_max_step_pos_m=cartesian_max_step_pos_m,
        cartesian_max_step_rot_rad=cartesian_max_step_rot_rad,
        jog_any_c_position_only=jog_any_c_position_only,
        jog_any_c_orientation_only=jog_any_c_orientation_only,
        jog_any_c_interrupt=jog_any_c_interrupt,
        zone_ratio=zone_ratio,
        jog_any_c_async_timeout_ms=jog_any_c_async_timeout_ms,
        cartesian_vel=cartesian_vel,
        cartesian_acc=cartesian_acc,
        cartesian_dec=cartesian_dec,
        joint_vel=joint_vel,
        joint_acc=joint_acc,
        joint_dec=joint_dec,
        jog_any_c_preview_only=jog_any_c_preview,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        enable_log_data=enable_log_data,
        log_dir=log_dir,
        log_freq=log_freq,
        enable_camera=enable_camera,
        camera_serial_dict=camera_serial_dict,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        enable_camera_depth=enable_camera_depth,
        enable_camera_compression=enable_camera_compression,
        camera_jpg_quality=camera_jpg_quality,
        log_joint_count=log_joint_count,
        gripper_trigger_name=gripper_trigger_name,
        gripper_trigger_threshold=gripper_trigger_threshold,
        gripper_open_cmd=gripper_open_cmd,
        gripper_closed_cmd=gripper_closed_cmd,
        gripper_observation_default=gripper_observation_default,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
