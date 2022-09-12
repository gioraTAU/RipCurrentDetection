import cv2
from pathlib import Path
import numpy as np

# thanks to: https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
class BoundingBoxWidget(object):
    def __init__(self, image, video_name):
        self.original_image = image#cv2.imread('1.jpg')
        self.clone = self.original_image.copy()
        self.video_name = video_name

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            top_left = self.image_coordinates[0]
            bottom_right = self.image_coordinates[1]

            xmin = top_left[0]
            ymin = bottom_right[1]
            xmax = bottom_right[0]
            ymax = top_left[1]
            data_list = [self.video_name]
            data_list.extend([str(x) for x in (xmin, ymin, xmax, ymax)])

            print(','.join(data_list))
            # print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            # print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1], self.image_coordinates[1][0] - self.image_coordinates[0][0], self.image_coordinates[1][1] - self.image_coordinates[0][1]))

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

base_path = Path('test_data')

# video_file_path = Path('test_data/rip_01.mp4')
for video_file_path in base_path.glob('rip_04*'):
    save_name = video_file_path.name.split('.')[0]


    RESIZE_TO = (500, 500)
    cap = cv2.VideoCapture(str(video_file_path))

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
    frame_counter = 0
    good_frames = []
    for i in range(100):
        # capture each frame of the video
        ret, frame = cap.read()
        # [{'frame: #, 'IOU': #}]

        if ret:
            image = cv2.resize(frame, RESIZE_TO)
            # image = frame.copy()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


    boundingbox_widget = BoundingBoxWidget(image, video_file_path.name)
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(1)
        if key == ord('n'):
            cv2.destroyAllWindows()
            break
