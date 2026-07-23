import qrcode
import json

def generate_turn_qr():
    # The instructions for the robot
    payload = {
        "action": "ROTATE_RIGHT",
        "speed": 0.0,         # Do not move forward
        "turn_speed": -0.5,   # Negative value spins the robot right
        "duration": 1.5       # Spin for 1.5 seconds
    }
    
    # Convert instructions to text
    json_str = json.dumps(payload)
    
    # Generate the QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(json_str)
    qr.make(fit=True)

    # Save the image
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("qr_turn_right.png")
    
    print("[+] Generated: qr_turn_right.png")
    print(f"    Payload: {json_str}")

if __name__ == "__main__":
    generate_turn_qr()