from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO('runs/pose/train2/weights/best.pt')

# Define custom skeleton connections based on keypoint indices
# Keypoint order: ['Bahu', 'Bahu Kanan', 'Jari', 'Kepala', 'Kepala 2', 'Kuping', 'Kuping Kanan', 'Lengan', 'Lutut Kanan', 'Lutut Kiri', 'Merokok', 'Paha', 'Pergelangan Kaki', 'Pergelangan Kaki Kanan', 'Pergelangan Tangan', 'Perut ke Dada', 'Pinggang', 'Sepatu', 'Siku', 'Siku Kiri']

custom_skeleton = [
    (3, 4),   # Kepala -> Kepala 2
    (3, 5),   # Kepala -> Kuping  
    (4, 6),   # Kepala 2 -> Kuping Kanan
    (3, 15),  # Kepala -> Perut ke Dada
    (15, 0),  # Perut ke Dada -> Bahu
    (15, 1),  # Perut ke Dada -> Bahu Kanan
    (0, 18),  # Bahu -> Siku
    (1, 19),  # Bahu Kanan -> Siku Kiri
    (18, 14), # Siku -> Pergelangan Tangan
    (19, 7),  # Siku Kiri -> Lengan
    (14, 2),  # Pergelangan Tangan -> Jari
    (14, 10), # Pergelangan Tangan -> Merokok (SMOKING CONNECTION!)
    (15, 16), # Perut ke Dada -> Pinggang
    (16, 11), # Pinggang -> Paha
    (11, 8),  # Paha -> Lutut Kanan
    (11, 9),  # Paha -> Lutut Kiri
    (8, 13),  # Lutut Kanan -> Pergelangan Kaki Kanan
    (9, 12),  # Lutut Kiri -> Pergelangan Kaki
    (12, 17), # Pergelangan Kaki -> Sepatu
    (13, 17)  # Pergelangan Kaki Kanan -> Sepatu
]

def draw_custom_skeleton(img, keypoints, connections, confidence_threshold=0.3):
    """Draw custom skeleton on image"""
    # Colors for different body parts
    colors = {
        'head': (0, 255, 255),      # Yellow
        'arms': (0, 255, 0),        # Green
        'smoking': (0, 0, 255),     # Red - SMOKING CONNECTION
        'body': (255, 0, 0),        # Blue
        'legs': (255, 255, 0)       # Cyan
    }
    
    # Define connection categories
    head_connections = [(3, 4), (3, 5), (4, 6)]
    arm_connections = [(0, 18), (1, 19), (18, 14), (19, 7), (14, 2)]
    smoking_connections = [(14, 10)]  # Hand to cigarette
    body_connections = [(3, 15), (15, 0), (15, 1), (15, 16), (16, 11)]
    leg_connections = [(11, 8), (11, 9), (8, 13), (9, 12), (12, 17), (13, 17)]
    
    for connection in connections:
        idx1, idx2 = connection
        
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            x1, y1, conf1 = keypoints[idx1]
            x2, y2, conf2 = keypoints[idx2]
            
            if conf1 > confidence_threshold and conf2 > confidence_threshold:
                # Choose color and thickness based on connection type
                if connection in smoking_connections:
                    color = colors['smoking']
                    thickness = 8  # Extra thick for smoking
                elif connection in head_connections:
                    color = colors['head']
                    thickness = 3
                elif connection in arm_connections:
                    color = colors['arms']
                    thickness = 4
                elif connection in body_connections:
                    color = colors['body']
                    thickness = 3
                elif connection in leg_connections:
                    color = colors['legs']
                    thickness = 3
                else:
                    color = (128, 128, 128)
                    thickness = 2
                
                # Draw the connection line
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return img

def draw_keypoints(img, keypoints, confidence_threshold=0.3):
    """Draw keypoints as colored circles with labels"""
    keypoint_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255)
    ]
    
    keypoint_names = ['Bahu', 'Bahu Kanan', 'Jari', 'Kepala', 'Kepala 2', 'Kuping', 
                     'Kuping Kanan', 'Lengan', 'Lutut Kanan', 'Lutut Kiri', 'Merokok', 
                     'Paha', 'Pergelangan Kaki', 'Pergelangan Kaki Kanan', 
                     'Pergelangan Tangan', 'Perut ke Dada', 'Pinggang', 'Sepatu', 
                     'Siku', 'Siku Kiri']
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold and i < len(keypoint_names):
            color = keypoint_colors[i % len(keypoint_colors)]
            
            # Special styling for smoking keypoint
            if i == 10:  # 'Merokok' keypoint
                radius = 8
                thickness = 4
                text_color = (0, 0, 255)
            else:
                radius = 6
                thickness = 2
                text_color = color
            
            # Draw keypoint circle
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), thickness)
            
            # Draw keypoint name
            label = keypoint_names[i]
            cv2.putText(img, label, (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

def is_smoking_detected(keypoints):
    """Check if smoking is detected based on keypoints"""
    if len(keypoints) > 10:  # Index 10 is 'Merokok'
        smoking_confidence = keypoints[10][2]
        return smoking_confidence > 0.5
    return False

def process_video(video_path, output_path=None):
    """Process video with smoking detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    smoking_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO inference
        results = model(frame, verbose=False)
        
        frame_has_smoking = False
        
        for result in results:
            # Process bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Check for smoking in this detection
                    detection_smoking = False
                    if result.keypoints is not None:
                        for keypoint_set in result.keypoints.data:
                            keypoints_xy = keypoint_set.cpu().numpy()
                            if is_smoking_detected(keypoints_xy):
                                detection_smoking = True
                                frame_has_smoking = True
                                break
                    
                    # Draw bounding box
                    if detection_smoking:
                        box_color = (0, 0, 255)  # Red for smoker
                        label = f"SMOKER {confidence:.2f}"
                    else:
                        box_color = (0, 255, 0)  # Green for person
                        label = f"person {confidence:.2f}"
                    
                    # Draw box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)), box_color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw skeleton and keypoints
            if result.keypoints is not None:
                for keypoint_set in result.keypoints.data:
                    keypoints_xy = keypoint_set.cpu().numpy()
                    
                    # Draw custom skeleton
                    frame = draw_custom_skeleton(frame, keypoints_xy, custom_skeleton)
                    
                    # Draw keypoints
                    draw_keypoints(frame, keypoints_xy)
        
        # Update smoking frame counter
        if frame_has_smoking:
            smoking_frames += 1
        
        # Add status text
        if frame_has_smoking:
            status_text = "SMOKING DETECTED!"
            status_color = (0, 0, 255)
        else:
            status_text = "No smoking detected"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count + 1}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Smoking Detection', frame)
        
        # Save frame
        if output_path:
            out.write(frame)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user")
            break
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    smoking_percentage = (smoking_frames / frame_count) * 100 if frame_count > 0 else 0
    
    print("\nPROCESSING COMPLETE")
    print(f"Total frames processed: {frame_count}")
    print(f"Smoking detected in: {smoking_frames} frames ({smoking_percentage:.1f}%)")
    if output_path:
        print(f"Output saved to: {output_path}")

def process_image(image_path, output_path=None):
    """Process single image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot open image {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # YOLO inference
    results = model(img, verbose=False)
    
    smoking_detected = False
    
    for result in results:
        # Process detections
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Check smoking
                if result.keypoints is not None:
                    for keypoint_set in result.keypoints.data:
                        keypoints_xy = keypoint_set.cpu().numpy()
                        if is_smoking_detected(keypoints_xy):
                            smoking_detected = True
                
                # Draw box
                box_color = (0, 0, 255) if smoking_detected else (0, 255, 0)
                label = f"SMOKER {confidence:.2f}" if smoking_detected else f"person {confidence:.2f}"
                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Draw skeleton
        if result.keypoints is not None:
            for keypoint_set in result.keypoints.data:
                keypoints_xy = keypoint_set.cpu().numpy()
                img = draw_custom_skeleton(img, keypoints_xy, custom_skeleton)
                draw_keypoints(img, keypoints_xy)
    
    # Show result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if needed
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")
    
    print(f"Smoking detected: {'YES' if smoking_detected else 'NO'}")

# Main execution
if __name__ == "__main__":
    print("Custom Smoking Pose Detection")
    print("=" * 40)
    
    # Simple menu
    print("1. Process video")
    print("2. Process image")
    print("3. Quick test (video1.mp4)")
    
    choice = input("Choose option [1/2/3]: ").strip()
    
    if choice == "1":
        video_path = input("Enter video path: ").strip()
        if not video_path:
            video_path = "video1.mp4"
        
        save_output = input("Save output? [y/n]: ").lower() == 'y'
        output_path = "output_smoking_detection.mp4" if save_output else None
        
        process_video(video_path, output_path)
        
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        save_output = input("Save output? [y/n]: ").lower() == 'y'
        output_path = "output_image.jpg" if save_output else None
        
        process_image(image_path, output_path)
        
    else:
        # Quick test
        print("Quick test with video1.mp4")
        process_video('video1.mp4', 'output_smoking_detection.mp4')
    
    print("\nDone! Thank you!")