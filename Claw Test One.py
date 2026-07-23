import time
import numpy as np

# Import Quanser hardware classes
from hal.products.qbot_platform import QBotPlatform
from hal.products.qarm_mini import QArmMini
from pal.resources.cameras import Camera2D

def operate_claw(arm, action):
    """Controls the QArm Mini's two-stage gripper."""
    if action == "OPEN":
        print("[-] Opening QArm Mini claw...")
        # 0.0 is typically the command for fully open
        arm.command_gripper(0.0)  
    elif action == "CLOSE":
        print("[+] Closing QArm Mini claw...")
        # 1.0 is typically the command for fully closed
        arm.command_gripper(1.0)  

def main():
    print("[INFO] Connecting to QBot hardware at IP host 71...")
    
    # 1. Initialize the base QBot platform
    qbot = QBotPlatform()
    
    # 2. Initialize the QArm Mini
    print("[INFO] Initializing QArm Mini...")
    arm = QArmMini()
    
    # 3. Initialize the RealSense Depth Camera
    print("[INFO] Initializing RealSense Depth Camera...")
    # Quanser's Camera2D class handles the depth map from the RealSense
    realsense_depth = Camera2D(camera_id="1", frame_width=640, frame_height=480, frame_rate=30)
    
    # Ensure claw is fully open before starting movement
    operate_claw(arm, "OPEN")
    time.sleep(1.0) 
    
    try:
        print("[INFO] Starting depth scanning loop. Driving forward...")
        while True:
            # Command QBot to drive forward slowly (0.1 meters per second)
            # arm=1 enables the motor drive
            qbot.write(arm=1, commands=np.array([0.1, 0.0], dtype=np.float64))
            
            # Read the 3D depth frame (pixels contain physical distance data)
            ret, depth_frame = realsense_depth.read_depth()
            
            if ret and depth_frame is not None:
                # Target the exact center pixel of the 640x480 camera view
                center_y, center_x = 240, 320
                distance_to_item = depth_frame[center_y, center_x]
                
                # Check if the item is closer than 15 cm (0.15m) and the reading is valid
                if 0.0 < distance_to_item < 0.15:
                    print(f"\n[!] Object detected at {distance_to_item:.3f} meters!")
                    print("[!] Stopping QBot wheels and engaging QArm Mini...")
                    
                    # 1. Immediately stop the robot's wheels
                    qbot.write(arm=1, commands=np.array([0.0, 0.0], dtype=np.float64))
                    time.sleep(0.5) # Give the chassis a moment to settle
                    
                    # 2. Trigger the claw to close and secure the item
                    operate_claw(arm, "CLOSE")
                    time.sleep(2.0) # Wait for the physical gripper to finish closing
                    
                    print("[INFO] Object secured. Ending automated sequence.")
                    break # Exit the loop since the task is complete
                    
            time.sleep(0.03) # Maintain a ~30 Hz loop rate
                    
    except KeyboardInterrupt:
        print("\n[INFO] Script manually stopped by user.")
    finally:
        # Safely shut down all hardware connections to prevent motor lockups
        print("[INFO] Shutting down hardware safely...")
        try:
            qbot.write(arm=0, commands=np.array([0.0, 0.0], dtype=np.float64))
        except Exception:
            pass
            
        realsense_depth.release()
        
        # Terminate QArm and QBot hardware handles
        try:
            arm.terminate()
        except AttributeError:
            pass
            
        qbot.terminate()

if __name__ == "__main__":
    main()