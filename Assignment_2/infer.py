# coding: utf-8

import os
import argparse

from ultralytics import YOLO
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for video object detection using pre-trained YOLO model.")

    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the video file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output video file.")
    parser.add_argument("-y", "--yolo", type=str, required=True, help="Path to the YOLO model.")

    return parser.parse_args()
    

def run_detection(
    yolo: YOLO,
    video_path: str,
    output_path: str,
) -> None:
    print("Creating video capture object ...")
    video_capture = cv2.VideoCapture(video_path)

    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not video_capture.isOpened():
        print(f"Error: Unable to open video file for reading {video_path}.")
        return
    print("Video capture object created.")
    
    print("Creating video writer object ...")
    output_file_path = os.path.join(output_path, "output.avi")
    video_writer = cv2.VideoWriter(
        output_file_path,
        cv2.VideoWriter_fourcc('M','J','P','G'), 
        frame_rate,
        (frame_width, frame_height),
    )

    if not video_writer.isOpened():
        print(f"Error: Unable to open video file for writing {output_file_path}.")
        return
    print("Video writer object created.")
    

    print("Detecting objects in the video...")
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            break
 
        result = yolo(frame)

        for object in result[0]:
            x1, y1, x2, y2 = object.boxes.xyxy.cpu().squeeze()
            label = int(object.boxes.cls.item())
            class_name = "mouse" if label == 0 else "keyboard"
            confidence = object.boxes.conf.item()

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0) if label == 0 else (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if label == 0 else (0, 0, 255),
                2,
            )

        video_writer.write(frame)

    video_capture.release()
    video_writer.release()
    print("Detection done.")
    

def main():
    args = parse_arguments()

    print("Loading YOLO model...")
    yolo = YOLO(args.yolo)
    print("Model loaded.")

    run_detection(
        yolo=yolo,
        video_path=args.video,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
