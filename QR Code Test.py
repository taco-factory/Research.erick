import json
import time
import numpy as np
import cv2
from pyzbar.pyzbar import decode

# Quanser Libraries
from hal.products.qbot_platform import QBotPlatform
from pal.resources.cameras import Camera2D
from pal.utilities.probe import Observer


def parse_qr_payload(data_str):
    """Parses incoming QR payload: supports JSON and colon-separated formats."""
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        pass

    parsed_data = {}
    try:
        parts = data_str.strip().split(";")
        for part in parts:
            if ":" in part:
                key, val = part.split(":")
                parsed_data[key] = val.strip()
        return parsed_data
    except Exception:
        return None


def process_qr_vision(frame):
    """
    Decodes QR codes in the image frame, draws bounding boxes,
    and returns detected node information along with the annotated frame.
    """
    if frame is None:
        return None, frame

    decoded_objects = decode(frame)
    detected_payload = None

    for obj in decoded_objects:
        raw_qr_str = obj.data.decode("utf-8")
        detected_payload = parse_qr_payload(raw_qr_str)

        # Draw green bounding polygon around detected QR code
        pts = obj.polygon
        if len(pts) > 0:
            pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

    return detected_payload, frame


def main():
    # -------------------------------------------------------------------------
    # 1. Hardware Initialization
    # -------------------------------------------------------------------------
    print("[INFO] Initializing QBot Platform Hardware...")
    qbot = QBotPlatform()

    # Initialize Downward Camera (camera_id '0', typically grayscale or RGB)
    down_cam = Camera2D(camera_id="0", frame_width=640, frame_height=400, frame_rate=30)

    # Initialize RealSense RGB Camera (camera_id '1')
    realsense_cam = Camera2D(camera_id="1", frame_width=640, frame_height=480, frame_rate=30)

    # -------------------------------------------------------------------------
    # 2. Observer Display Setup
    # -------------------------------------------------------------------------
    observer = Observer()
    observer.add_display(
        imageSize=[400, 640, 1],
        scalingFactor=2,
        name='Downward Facing Image'
    )
    observer.add_display(
        imageSize=[480, 640, 3],
        scalingFactor=2,
        name='RealSense RGB Image'
    )
    observer.add_display(
        imageSize=[480, 640, 1],
        scalingFactor=2,
        name='RealSense Depth Image'
    )
    observer.add_plot(
        numMeasurements=1680,
        frameSize=400,
        pixelsPerMeter=50,
        scalingFactor=8,
        name='Leishen M10P Lidar'
    )
    observer.launch()

    print("[INFO] System online. Scanning for QR codes...")

    # -------------------------------------------------------------------------
    # 3. Main Control Loop
    # -------------------------------------------------------------------------
    current_node = None
    
    # Motor command speeds [forward_velocity (m/s), turn_rate (rad/s)]
    cmd_forward = 0.0
    cmd_turn = 0.0

    try:
        while True:
            # Read camera frames
            ret_down, frame_down = down_cam.read()
            ret_rs, frame_rs = realsense_cam.read()

            # --- Process Downward Camera & Scan QR Codes ---
            if ret_down and frame_down is not None:
                node_info, annotated_down = process_qr_vision(frame_down)

                if node_info and node_info != current_node:
                    current_node = node_info
                    print("\n[+] NEW QR NAVIGATION NODE DETECTED:")
                    print(f"    Raw / Parsed Data: {current_node}")

                    # Determine motion behavior based on QR payload parameters
                    action = current_node.get("action") or current_node.get("DIR")

                    if action in ["FORWARD", "N"]:
                        cmd_forward = float(current_node.get("speed", 0.15))
                        cmd_turn = 0.0
                        print("    -> Action Executed: Drive Forward")

                    elif action in ["ROTATE_LEFT", "L"]:
                        cmd_forward = 0.0
                        cmd_turn = 0.5  # Positive turn rate for left rotation
                        print("    -> Action Executed: Rotate Left")

                    elif action in ["ROTATE_RIGHT", "R"]:
                        cmd_forward = 0.0
                        cmd_turn = -0.5  # Negative turn rate for right rotation
                        print("    -> Action Executed: Rotate Right")

                    elif action == "STOP":
                        cmd_forward = 0.0
                        cmd_turn = 0.0
                        print("    -> Action Executed: Stop")

                # Format frame for Observer display (ensuring single channel for grayscale)
                if len(annotated_down.shape) == 3:
                    annotated_down = cv2.cvtColor(annotated_down, cv2.COLOR_BGR2GRAY)
                
                observer.update_display('Downward Facing Image', annotated_down)

            # --- Process RealSense Camera ---
            if ret_rs and frame_rs is not None:
                observer.update_display('RealSense RGB Image', frame_rs)

            # --- Send Motor Commands to QBot ---
            # arm = 1 enables motor drive outputs
            # commands = np.array([forward_speed, turn_speed], dtype=np.float64)
            commands = np.array([cmd_forward, cmd_turn], dtype=np.float64)
            # qbot.write(arm=1, commands=commands)  # Uncomment when sending commands to physical motors

            time.sleep(0.03)  # Maintain ~30 Hz execution frequency

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected. Stopping robot...")

    finally:
        # -------------------------------------------------------------------------
        # 4. Safe Hardware Termination
        # -------------------------------------------------------------------------
        # Emergency stop motors
        try:
            qbot.write(arm=0, commands=np.array([0.0, 0.0], dtype=np.float64))
        except Exception:
            pass

        down_cam.release()
        realsense_cam.release()
        qbot.terminate()
        print("[INFO] Hardware, cameras, and Observer displays safely terminated.")


if __name__ == "__main__":
    main()