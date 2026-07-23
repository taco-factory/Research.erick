import time
import numpy as np

# Import Quanser hardware classes
from hal.products.qbot_platform import QBotPlatform
from hal.products.qarm_mini import QArmMini
from pal.resources.cameras import Camera2D

def set_gripper(arm, action):
    """Safely commands the QArm Mini gripper."""
    try:
        if action == "OPEN":
            print("[-] Actuating QArm Mini: OPENING...")
            arm.command_gripper(0.0)  # 0.0 = Open
        elif action == "CLOSE":
            print("[+] Actuating QArm Mini: CLOSING...")
            arm.command_gripper(1.0)  # 1.0 = Closed
    except Exception as e:
        print(f"[ERROR] Failed to actuate gripper: {e}")

def main():
    print("[INFO] Initializing QBot Platform and QArm Mini...")
    
    # Initialize hardware connections
    qbot = QBotPlatform()
    arm = QArmMini()
    realsense_depth = Camera2D(camera_id="1", frame_width=640, frame_height=480, frame_rate=30)
    
    # Ensure claw is open before starting
    set_gripper(arm, "OPEN")
    time.sleep(1.0) 
    
    # Define system states
    robot_state = "SEARCHING"
    
    print("[INFO] System Ready. Beginning autonomous search...")

    try:
        while True:
            if robot_state == "SEARCHING":
                # Drive forward slowly (0.1 m/s)
                qbot.write(arm=1, commands=np.array([0.1, 0.0], dtype=np.float64))
                
                # Fetch depth frame
                ret, depth_frame = realsense_depth.read_depth()
                
                if ret and depth_frame is not None:
                    # Safely check the center pixel distance
                    try:
                        center_y, center_x = 240, 320
                        distance_to_item = depth_frame[center_y, center_x]
                        
                        # If an object is detected between 1cm and 15cm away
                        if 0.01 < distance_to_item < 0.15:
                            print(f"\n[!] Target acquired at {distance_to_item:.3f} meters!")
                            robot_state = "GRABBING"
                            
                    except IndexError:
                        print("[WARNING] Depth frame dimensions are incorrect. Skipping frame.")
            
            elif robot_state == "GRABBING":
                print("[INFO] Halting drive motors...")
                # Stop the wheels
                qbot.write(arm=1, commands=np.array([0.0, 0.0], dtype=np.float64))
                time.sleep(1.0) # Allow chassis momentum to settle
                
                # Execute grab
                set_gripper(arm, "CLOSE")
                time.sleep(2.0) # Wait for servo to finish closing
                
                print("[INFO] Payload secured. Task complete.")
                break # Exit the while loop
                
            # Run loop at roughly 30 Hz
            time.sleep(0.03) 
                    
    except KeyboardInterrupt:
        print("\n[INFO] Manual override detected. Halting sequence.")
        
    finally:
        print("[INFO] Executing safe hardware shutdown...")
        # 1. Kill motor power
        try:
            qbot.write(arm=0, commands=np.array([0.0, 0.0], dtype=np.float64))
        except Exception:
            pass
            
        # 2. Release camera
        realsense_depth.release()
        
        # 3. Terminate arm and base
        try:
            arm.terminate()
        except AttributeError:
            pass
            
        qbot.terminate()
        print("[INFO] System powered down successfully.")

if __name__ == "__main__":
    main()