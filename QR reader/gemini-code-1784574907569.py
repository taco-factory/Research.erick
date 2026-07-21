import json
import time
import cv2
from pyzbar.pyzbar import decode

# Import Quanser hardware/PAL libraries from Quanser_Academic_Resources
from pal.resources.cameras import Camera2D
from hal.products.qbot_platform import QBotPlatform


def parse_qr_payload(data_str):
    """Parses QR string into structured coordinate data."""
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        pass

    # Fallback for delimited strings (e.g. NODE:001;X:0;Y:0;DIR:N)
    parsed_data = {}
    try:
        parts = data_str.strip().split(";")
        for part in parts:
            if ":" in part:
                key, val = part.split(":")
                parsed_data[key] = val
        return parsed_data
    except Exception:
        return None


def main():
    # 1. Initialize QBot Platform Hardware / Drivers
    qbot = QBotPlatform()
    
    # 2. Initialize Downward/Front Camera using Quanser Camera2D class
    # camera_id: '0' for primary down-facing camera or USB cam feed on QBot
    cam = Camera2D(camera_id="0", frame_width=640, frame_height=480, frame_rate=30)

    print("[INFO] Initializing QBot Platform Sensors & Camera...")
    
    current_node = None

    try:
        while True:
            # Read a fresh image frame from Quanser Camera API
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # Detect and decode QR codes in frame
            decoded_objects = decode(frame)

            for obj in decoded_objects:
                qr_data = obj.data.decode("utf-8")
                node_info = parse_qr_payload(qr_data)

                if node_info and node_info != current_node:
                    current_node = node_info
                    print(f"\n[+] NAV NODE DETECTED:")
                    print(f"    Raw Payload : {qr_data}")
                    print(f"    Parsed Data : {node_info}")

                    # Example: Access individual grid values for navigation
                    # x_pos = node_info.get("x")
                    # y_pos = node_info.get("y")

                # Visual bounding box on camera feed
                pts = obj.polygon
                if len(pts) > 0:
                    for i in range(len(pts)):
                        cv2.line(
                            frame,
                            pts[i],
                            pts[(i + 1) % len(pts)],
                            (0, 255, 0),
                            2,
                        )

            # Display video feed
            cv2.imshow("QBot Platform Vision Feed", frame)

            # Press 'q' on keyboard to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.03)  # Loop delay (~30 FPS)

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted program.")

    finally:
        # Clean up camera and QBot hardware handles safely
        cam.release()
        cv2.destroyAllWindows()
        qbot.terminate()
        print("[INFO] QBot hardware session closed gracefully.")


if __name__ == "__main__":
    main()