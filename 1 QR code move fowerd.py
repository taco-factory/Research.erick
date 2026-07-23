import qrcode
import json

def generate_forward_qr():
    # The instructions for the robot
    payload = {
        "action": "FORWARD",
        "speed": 0.22,      # Meters per second
        "duration": 7.0     # Run for 7 seconds
    }
    
    # Convert instructions to text
    json_str = json.dumps(payload)
    
    # Generate the QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(json_str)
    qr.make(fit=True)

    # Save the image
    img = qr.make_image(fill_color="black", back_color="white")
    img.save("qr_forward.png")
    
    print("[+] Generated: qr_forward.png")
    print(f"    Payload: {json_str}")

if __name__ == "__main__":
    generate_forward_qr()