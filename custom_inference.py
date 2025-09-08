from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained model
model = YOLO('runs/pose/train/weights/best.pt')

# CONFIGURATION SECTION - Customize for your specific use case
# =============================================================

# Define your custom keypoint names (modify based on your annotation)
# Example for smoking detection: ['Head', 'Left_Shoulder', 'Right_Shoulder', 'Left_Wrist', 'Right_Wrist', 'Cigarette', ...]
KEYPOINT_NAMES = [
    'Shoulder', 'Right_Shoulder', 'Fingers', 'Head', 'Head_2', 'Left_Ear', 
    'Right_Ear', 'Arm', 'Right_Knee', 'Left_Knee', 'Special_Object',  # <- Key object for detection (e.g., cigarette, tool, etc.)
    'Thigh', 'Ankle', 'Right_Ankle', 'Wrist', 'Chest', 'Waist', 'Shoe', 
    'Elbow', 'Left_Elbow'
]

# Define custom skeleton connections based on keypoint indices
# Format: (keypoint1_index, keypoint2_index)
CUSTOM_SKELETON = [
    (3, 4),   # Head -> Head_2
    (3, 5),   # Head -> Left_Ear  
    (4, 6),   # Head_2 -> Right_Ear
    (3, 15),  # Head -> Chest
    (15, 0),  # Chest -> Shoulder
    (15, 1),  # Chest -> Right_Shoulder
    (0, 18),  # Shoulder -> Elbow
    (1, 19),  # Right_Shoulder -> Left_Elbow
    (18, 14), # Elbow -> Wrist
    (19, 7),  # Left_Elbow -> Arm
    (14, 2),  # Wrist -> Fingers
    (14, 10), # Wrist -> Special_Object (KEY CONNECTION for behavior detection)
    (15, 16), # Chest -> Waist
    (16, 11), # Waist -> Thigh
    (11, 8),  # Thigh -> Right_Knee
    (11, 9),  # Thigh -> Left_Knee
    (8, 13),  # Right_Knee -> Right_Ankle
    (9, 12),  # Left_Knee -> Ankle
    (12, 17), # Ankle -> Shoe
    (13, 17)  # Right_Ankle -> Shoe
]

# Define color scheme for different body parts
CONNECTION_COLORS = {
    'head': (0, 255, 255),        # Yellow - head connections
    'arms': (0, 255, 0),          # Green - arm connections
    'special': (0, 0, 255),       # Red - special object connection (key for behavior detection)
    'body': (255, 0, 0),          # Blue - body/torso connections
    'legs': (255, 255, 0)         # Cyan - leg connections
}

# Define which connections belong to which category
CONNECTION_CATEGORIES = {
    'head': [(3, 4), (3, 5), (4, 6)],
    'arms': [(0, 18), (1, 19), (18, 14), (19, 7), (14, 2)],
    'special': [(14, 10)],  # Hand to special object - customize this for your use case
    'body': [(3, 15), (15, 0), (15, 1), (15, 16), (16, 11)],
    'legs': [(11, 8), (11, 9), (8, 13), (9, 12), (12, 17), (13, 17)]
}

# Behavior detection configuration
SPECIAL_KEYPOINT_INDEX = 10  # Index of your special object/behavior keypoint
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence for keypoint detection
BEHAVIOR_THRESHOLD = 0.5     # Threshold for behavior classification

# Classification labels
POSITIVE_LABEL = "BEHAVIOR_DETECTED"     # Label when behavior is detected (e.g., "SMOKING", "WORKING", "PLAYING")
NEGATIVE_LABEL = "NO_BEHAVIOR"           # Label when no behavior detected
POSITIVE_CLASS_NAME = "person_with_behavior"  # Class name for positive detection
NEGATIVE_CLASS_NAME = "person"           # Class name for negative detection

# =============================================================

def draw_custom_skeleton(img, keypoints, connections, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw custom skeleton connections on image"""
    
    for connection in connections:
        idx1, idx2 = connection
        
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            x1, y1, conf1 = keypoints[idx1]
            x2, y2, conf2 = keypoints[idx2]
            
            if conf1 > confidence_threshold and conf2 > confidence_threshold:
                # Determine color and thickness based on connection category
                color = (128, 128, 128)  # Default gray
                thickness = 2
                
                for category, category_connections in CONNECTION_CATEGORIES.items():
                    if connection in category_connections:
                        color = CONNECTION_COLORS[category]
                        if category == 'special':
                            thickness = 8  # Extra thick for special connection
                        elif category == 'arms':
                            thickness = 4
                        else:
                            thickness = 3
                        break
                
                # Draw the connection line
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return img

def draw_keypoints(img, keypoints, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Draw keypoints as colored circles with labels"""
    keypoint_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255)
    ]
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold and i < len(KEYPOINT_NAMES):
            color = keypoint_colors[i % len(keypoint_colors)]
            
            # Special styling for the key behavior keypoint
            if i == SPECIAL_KEYPOINT_INDEX:
                radius = 8
                thickness = 4
                text_color = (0, 0, 255)  # Red for special object
            else:
                radius = 6
                thickness = 2
                text_color = color
            
            # Draw keypoint circle
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), thickness)
            
            # Draw keypoint label
            if i < len(KEYPOINT_NAMES):
                label = KEYPOINT_NAMES[i]
                cv2.putText(img, label, (int(x) + 10, int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

def is_behavior_detected(keypoints):
    """
    Check if the target behavior is detected based on keypoints
    Customize this function based on your specific use case
    
    Examples:
    - Smoking detection: Check if hand is near mouth and cigarette is detected
    - Tool usage: Check if tool keypoint is active and hand position
    - Activity detection: Check pose configuration for specific activity
    """
    if len(keypoints) > SPECIAL_KEYPOINT_INDEX:
        special_confidence = keypoints[SPECIAL_KEYPOINT_INDEX][2]
        return special_confidence > BEHAVIOR_THRESHOLD
    return False

def analyze_pose_behavior(keypoints):
    """
    Advanced behavior analysis based on pose configuration
    Customize this function for more sophisticated behavior detection
    
    Returns:
        dict: Analysis results with confidence scores and metrics
    """
    results = {
        'behavior_detected': False,
        'confidence': 0.0,
        'distance_metric': 0.0,
        'pose_score': 0.0
    }
    
    # Check special keypoint
    if is_behavior_detected(keypoints):
        results['behavior_detected'] = True
        results['confidence'] = keypoints[SPECIAL_KEYPOINT_INDEX][2]
    
    # Example: Calculate distance between hand and head for behavior analysis
    # Customize indices based on your keypoint configuration
    if len(keypoints) > 14 and len(keypoints) > 3:  # Wrist and Head indices
        hand_pos = keypoints[14][:2]  # Wrist position
        head_pos = keypoints[3][:2]   # Head position
        distance = np.linalg.norm(np.array(hand_pos) - np.array(head_pos))
        results['distance_metric'] = distance
        
        # Example scoring based on distance (customize threshold for your use case)
        if distance < 100:  # Close proximity threshold
            results['pose_score'] = min(1.0, (100 - distance) / 100)
    
    return results

def process_video(video_path, output_path=None):
    """Process video with custom behavior detection"""
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
    behavior_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO inference
        results = model(frame, verbose=False)
        
        frame_has_behavior = False
        
        for result in results:
            # Process bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Check for behavior in this detection
                    detection_behavior = False
                    if result.keypoints is not None:
                        for keypoint_set in result.keypoints.data:
                            keypoints_xy = keypoint_set.cpu().numpy()
                            if is_behavior_detected(keypoints_xy):
                                detection_behavior = True
                                frame_has_behavior = True
                                break
                    
                    # Draw bounding box with appropriate label
                    if detection_behavior:
                        box_color = (0, 0, 255)  # Red for positive behavior
                        label = f"{POSITIVE_CLASS_NAME.upper()} {confidence:.2f}"
                    else:
                        box_color = (0, 255, 0)  # Green for normal person
                        label = f"{NEGATIVE_CLASS_NAME} {confidence:.2f}"
                    
                    # Draw bounding box
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
                    frame = draw_custom_skeleton(frame, keypoints_xy, CUSTOM_SKELETON)
                    
                    # Draw keypoints
                    draw_keypoints(frame, keypoints_xy)
        
        # Update behavior frame counter
        if frame_has_behavior:
            behavior_frames += 1
        
        # Add status text
        if frame_has_behavior:
            status_text = POSITIVE_LABEL
            status_color = (0, 0, 255)
        else:
            status_text = NEGATIVE_LABEL
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count + 1}/{total_frames}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Custom Behavior Detection', frame)
        
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
    behavior_percentage = (behavior_frames / frame_count) * 100 if frame_count > 0 else 0
    
    print("\nPROCESSING COMPLETE")
    print(f"Total frames processed: {frame_count}")
    print(f"Behavior detected in: {behavior_frames} frames ({behavior_percentage:.1f}%)")
    if output_path:
        print(f"Output saved to: {output_path}")

def process_image(image_path, output_path=None):
    """Process single image with behavior detection"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot open image {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # YOLO inference
    results = model(img, verbose=False)
    
    behavior_detected = False
    
    for result in results:
        # Process detections
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Check behavior
                if result.keypoints is not None:
                    for keypoint_set in result.keypoints.data:
                        keypoints_xy = keypoint_set.cpu().numpy()
                        if is_behavior_detected(keypoints_xy):
                            behavior_detected = True
                
                # Draw bounding box
                box_color = (0, 0, 255) if behavior_detected else (0, 255, 0)
                label = f"{POSITIVE_CLASS_NAME.upper()} {confidence:.2f}" if behavior_detected else f"{NEGATIVE_CLASS_NAME} {confidence:.2f}"
                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                cv2.putText(img, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Draw skeleton and keypoints
        if result.keypoints is not None:
            for keypoint_set in result.keypoints.data:
                keypoints_xy = keypoint_set.cpu().numpy()
                img = draw_custom_skeleton(img, keypoints_xy, CUSTOM_SKELETON)
                draw_keypoints(img, keypoints_xy)
    
    # Show result
    cv2.imshow('Behavior Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save if needed
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Result saved to: {output_path}")
    
    print(f"Behavior detected: {'YES' if behavior_detected else 'NO'}")

# Main execution
if __name__ == "__main__":
    print("Dynamic Custom Skeleton Pose Detection")
    print("=" * 50)
    print("This tool can be customized for various behavior detection tasks:")
    print("- Smoking detection")
    print("- Tool usage monitoring") 
    print("- Activity recognition")
    print("- Safety compliance checking")
    print("- Sports movement analysis")
    print("- And more...")
    print("=" * 50)
    
    # Configuration info
    print(f"\nCurrent Configuration:")
    print(f"- Special keypoint: {KEYPOINT_NAMES[SPECIAL_KEYPOINT_INDEX]} (index {SPECIAL_KEYPOINT_INDEX})")
    print(f"- Behavior threshold: {BEHAVIOR_THRESHOLD}")
    print(f"- Total keypoints: {len(KEYPOINT_NAMES)}")
    print(f"- Skeleton connections: {len(CUSTOM_SKELETON)}")
    
    # Menu
    print("\nChoose processing option:")
    print("1. Process video")
    print("2. Process image")
    print("3. Quick test mode")
    
    choice = input("\nEnter your choice [1/2/3]: ").strip()
    
    if choice == "1":
        video_path = input("Enter video path: ").strip()
        if not video_path:
            video_path = "test_video.mp4"
            print(f"Using default: {video_path}")
        
        save_output = input("Save output video? [y/n]: ").lower() == 'y'
        output_path = "output_behavior_detection.mp4" if save_output else None
        
        process_video(video_path, output_path)
        
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        if not image_path:
            print("Image path is required!")
        else:
            save_output = input("Save output image? [y/n]: ").lower() == 'y'
            output_path = "output_behavior_detection.jpg" if save_output else None
            
            process_image(image_path, output_path)
        
    else:
        # Quick test mode
        print("Quick test mode - processing default video")
        process_video('test_video.mp4', 'output_behavior_detection.mp4')
    
    print("\nProcessing complete!")
    print("\nTo customize this tool for your specific use case:")
    print("1. Modify KEYPOINT_NAMES to match your annotation")
    print("2. Update CUSTOM_SKELETON connections")
    print("3. Adjust SPECIAL_KEYPOINT_INDEX for your key behavior")
    print("4. Customize is_behavior_detected() function")
    print("5. Update labels and thresholds as needed")