"""Preview RealSense color + depth streams. Requires opencv-python (with GUI), not headless."""

import cv2
import numpy as np

from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface


def _require_opencv_gui() -> None:
    if not hasattr(cv2, "imshow") or not hasattr(cv2, "waitKey"):
        raise RuntimeError(
            "OpenCV GUI 不可用（缺少 imshow/waitKey）。\n"
            "请安装带界面的 opencv-python，并确保未安装 opencv-python-headless：\n"
            "  pip uninstall opencv-python-headless -y\n"
            "  pip install opencv-python"
        )
    gui_line = next((l for l in cv2.getBuildInformation().splitlines() if "GUI:" in l), "")
    if "NONE" in gui_line:
        raise RuntimeError(f"当前 OpenCV 无 GUI 支持（{gui_line.strip()}），请改用 opencv-python。")


def main():
    _require_opencv_gui()
    windows_created: set[str] = set()

    try:
        with RealSenseCameraInterface() as camera_interface:
            # Warm up: first frames after start can exceed 500 ms.
            for _ in range(30):
                camera_interface.update_frames()
                if camera_interface.get_frames():
                    break

            while True:
                camera_interface.update_frames()
                frames_dict = camera_interface.get_frames()
                if not frames_dict:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                for serial, frames in frames_dict.items():
                    color_image = cv2.cvtColor(frames["color"], cv2.COLOR_RGB2BGR)
                    depth_image = frames.get("depth")

                    if depth_image is not None:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                        )
                        images = np.hstack((color_image, depth_colormap))
                    else:
                        images = color_image

                    win = f"RealSense - {serial}"
                    if win not in windows_created:
                        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
                        windows_created.add(win)
                    cv2.imshow(win, images)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except RuntimeError as e:
        print(e)
    finally:
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
