"""Verify TB6-R5 / TB5-R6 robot SDK binaries and a minimal RPC round-trip.

tb5r6 is a LeRobot config alias for tb6r5; both use the same RPC/topic libraries.

Usage:
    python scripts/hardware/verify_tb6r5_sdk.py
    python scripts/hardware/verify_tb6r5_sdk.py --robot-ip 192.168.11.11 --send-test-cmd
"""

from __future__ import annotations

import platform
import sys

import tyro

from xrobotoolkit_teleop.hardware.interface.tb6r5 import (
    TB6R5Interface,
    _host_elf_machine,
    _platform_subdir,
    _rpc_lib_dir,
    _setup_rpc_import,
    _topic_lib_dir,
    validate_robot_sdk,
)


def main(
    robot_ip: str = "192.168.11.11",
    rpc_port: int = 5868,
    send_test_cmd: bool = False,
    topic_only: bool = False,
) -> None:
    subdir = _platform_subdir()
    host = _host_elf_machine()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Platform: {platform.system()} {platform.machine()} (ELF {host}, sdk subdir={subdir})")
    print(f"Python: {py_ver}")
    print(f"RPC lib dir:   {_rpc_lib_dir()}")
    print(f"Topic lib dir: {_topic_lib_dir()}")

    validate_robot_sdk(require_topic=not topic_only)
    print("SDK file check: OK")

    if topic_only:
        arm = TB6R5Interface(ip=robot_ip, rpc_port=rpc_port, enable_topic=True)
        arm.connect_topic_feedback(topic_wait_timeout_s=5.0)
        q = arm.get_joint_positions()
        print(f"Topic feedback: OK (joint sample={q[:3].tolist()} ...)")
        arm.disconnect_topic_feedback()
        return

    arm = TB6R5Interface(ip=robot_ip, rpc_port=rpc_port, enable_topic=True)
    if send_test_cmd:
        arm.connect(topic_wait_timeout_s=5.0)
        print("RPC connect + init: OK")
        arm.disconnect()
        return

    # Dry connectivity check without init commands when robot may be offline.
    validate_robot_sdk(require_topic=True)
    _setup_rpc_import()
    import rpc

    print(f"import rpc: OK ({rpc})")
    client = rpc.CPPClient(robot_ip, rpc_port)
    del client
    print(f"CPPClient({robot_ip}, {rpc_port}): OK (no Enable/Start sent; pass --send-test-cmd to init)")


if __name__ == "__main__":
    tyro.cli(main)
