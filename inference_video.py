import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='test_data/rip_01.mp4',
    default='test_data/rip_01.mp4'
)
args = vars(parser.parse_args())
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.6
RESIZE_TO = (500, 500)
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      RESIZE_TO)
frame_count = 0  # to count total frames
total_fps = 0  # to get the final frames per second
# read until end of video
cap.set(1, 1)
N_agg = 60  # number of frames to aggregate
T_val = 40  # threshold value to extract aggregated rectangle
frame_agg = np.zeros((N_agg, 300, 300))  # aggregation frames
n_frame = 0
while cap.isOpened():
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, RESIZE_TO)
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).to(DEVICE)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1

        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            if boxes.shape[0] > 1:
                S = (boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0])
                boxes = boxes[np.argmax(S), :]
                boxes = np.reshape(boxes, (1, 4))
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            if boxes.shape[0] > 0:

                frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], boxes[0, 0]:boxes[0, 2]] = frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], boxes[0, 0]:boxes[0, 2]] + 1

                frame_agg[n_frame % N_agg, 0:boxes[0, 1], :] = frame_agg[n_frame % N_agg, 0:boxes[0, 1], :] - 1
                frame_agg[n_frame % N_agg, boxes[0, 3]:, :] = frame_agg[n_frame % N_agg, boxes[0, 3]:, :] - 1
                frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], 0:boxes[0, 0]] = frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], 0:boxes[0, 0]] - 1
                frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], boxes[0, 2]:] = frame_agg[n_frame % N_agg, boxes[0, 1]:boxes[0, 3], boxes[0, 2]:] - 1

            if n_frame > N_agg:
                frame_agg_cum = np.sum(frame_agg, 0)
                frame_agg_cum[frame_agg_cum < 0] = 0
                frame_agg_cum[frame_agg_cum > N_agg] = N_agg

                ids_rect = np.where(frame_agg_cum > T_val)
                ys = ids_rect[0]
                xs = ids_rect[1]

                if ids_rect[0].shape[0] > 0:

                    agg_boxes = np.array((min(xs), min(ys), max(xs), max(ys)))
                    draw_boxes = agg_boxes
                    draw_boxes = np.reshape(draw_boxes, (1, 4))

                # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, 2)
                cv2.putText(frame, class_name,
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)
        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        n_frame += 1
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")