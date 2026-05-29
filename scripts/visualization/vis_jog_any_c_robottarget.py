"""PICO JogAnyC preview: same teleop math as hardware, Meshcat only, no RPC.

Uses the full TB6R5 jog_any_c pipeline (PICO grip + delta -> robottarget) but never
connects to the robot. TCP reference comes from Placo FK at home pose (URDF ee_Link).

Usage:
    python scripts/visualization/vis_jog_any_c_robottarget.py
    python scripts/visualization/vis_jog_any_c_robottarget.py --no-jog-any-c-position-only
    python scripts/visualization/vis_jog_any_c_robottarget.py --jog-any-c-orientation-only
    python scripts/visualization/vis_jog_any_c_robottarget.py --scale-factor 1.5 --jog-any-c-interrupt on

Meshcat frames (same as hardware --visualize-placo):
    jog_any_c/right_hand/01_topic_tcp      current TCP (Placo FK in preview)
    jog_any_c/right_hand/02_current_tcp    latched ref at grip press
    jog_any_c/right_hand/03_target_to_send   computed JogAnyC robottarget
    viz_controller_scaled/right_hand       scaled PICO controller frame

Terminal prints the JogAnyC command string that would be sent (no RPC).
"""

import tyro

from xrobotoolkit_teleop.hardware.tb6r5_teleop_controller import (
    TB6R5TeleopController,
    DEFAULT_URDF_PATH,
    DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    DEFAULT_SCALE_FACTOR,
    DEFAULT_CARTESIAN_MAX_STEP_POS_M,
    DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD,
    DEFAULT_JOG_ANY_C_POSITION_ONLY,
    DEFAULT_JOG_ANY_C_ORIENTATION_ONLY,
    DEFAULT_JOG_ANY_C_INTERRUPT,
    DEFAULT_ZONE_RATIO,
    JogAnyCInterruptMode,
)


def main(
    robot_urdf_path: str = DEFAULT_URDF_PATH,
    manipulator_config: dict = DEFAULT_TB6R5_MANIPULATOR_CONFIG,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    cartesian_max_step_pos_m: float = DEFAULT_CARTESIAN_MAX_STEP_POS_M,
    cartesian_max_step_rot_rad: float = DEFAULT_CARTESIAN_MAX_STEP_ROT_RAD,
    jog_any_c_position_only: bool = DEFAULT_JOG_ANY_C_POSITION_ONLY,
    jog_any_c_orientation_only: bool = DEFAULT_JOG_ANY_C_ORIENTATION_ONLY,
    jog_any_c_interrupt: JogAnyCInterruptMode = DEFAULT_JOG_ANY_C_INTERRUPT,
    zone_ratio: float = DEFAULT_ZONE_RATIO,
    cartesian_vel: float | None = None,
    cartesian_acc: float | None = None,
    cartesian_dec: float | None = None,
    control_rate_hz: int = 50,
) -> None:
    controller = TB6R5TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=manipulator_config,
        robot_ip="none",
        teleop_mode="jog_any_c",
        jog_any_c_preview_only=True,
        scale_factor=scale_factor,
        cartesian_max_step_pos_m=cartesian_max_step_pos_m,
        cartesian_max_step_rot_rad=cartesian_max_step_rot_rad,
        jog_any_c_position_only=jog_any_c_position_only,
        jog_any_c_orientation_only=jog_any_c_orientation_only,
        jog_any_c_interrupt=jog_any_c_interrupt,
        zone_ratio=zone_ratio,
        cartesian_vel=cartesian_vel,
        cartesian_acc=cartesian_acc,
        cartesian_dec=cartesian_dec,
        visualize_placo=True,
        control_rate_hz=control_rate_hz,
        enable_log_data=False,
        enable_camera=False,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
