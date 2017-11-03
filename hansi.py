import numpy as np
import cv2
import json
import uuid
import time
import datetime
import pika

json.JSONEncoder.default = lambda self, obj: (obj.isoformat() if isinstance(obj, datetime.datetime) else None)

VIDEO_SOURCE_INDEX = 3

REGION_OF_INTEREST = {
    "x": 300,
    "y": 810,
    "width": 460,
    "height": 610
}

MIN_DETECTIONS_TILL_PERCEIVED_AS_BOTTLE = 50
DETECTION_MISSES_TILL_REMOVED = 20
MESSAGE_SENDING_THRESHOLD_BOTTLE_COUNT = 2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

SSD_PROTO_FILE = "./MobileNetSSD_deploy.prototxt.txt"
SSD_MODEL_FILE = "./MobileNetSSD_deploy.caffemodel"


class CameraCapture:
    def __init__(self):
        self.capture = None
        self.start(source_number=VIDEO_SOURCE_INDEX)

    def start(self, source_number):
        capture = cv2.VideoCapture(source_number)

        if not capture.isOpened():
            raise Exception("Unable to read camera feed")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.capture = capture

    def stop(self):
        self.capture.release()

    def get_current_image_rotated(self):
        success, image = self.capture.read()
        if not success:
            raise Exception("Could not get image")
        image = cv2.flip(cv2.transpose(image), 0)
        return image


def detect_bottles_with_ssd(image):
    network = cv2.dnn.readNetFromCaffe(SSD_PROTO_FILE, SSD_MODEL_FILE)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    network.setInput(blob)
    detections = network.forward()

    num_detections = 0

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        if i > 1000:
            break

        try:
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.1:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)

                cv2.rectangle(image, (startX, startY), (endX, endY),
                              COLORS[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15

                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                num_detections += 1

        except Exception as e:
            print("could not get label", e.message)

    display_image("bottom", image)

    return num_detections, image


def detect_bottles_with_hough_transform(detected_circles, image, top_image):
    matched_circles = list()

    num_bottles_top = 0

    # detect circles in the image
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, minDist=20, param1=22, param2=35, minRadius=7,
                               maxRadius=18)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            is_known_circle = False

            for (x_d, y_d, misses, detections) in detected_circles:
                i = detected_circles.index((x_d, y_d, misses, detections))

                match_x = is_within(x, x_d - r, x_d + r)
                match_y = is_within(y, y_d - r, y_d + r)

                if match_x and match_y:
                    detections.append((x, y, r))
                    matched_circles.append(i)
                    if len(detections) > MIN_DETECTIONS_TILL_PERCEIVED_AS_BOTTLE:
                        del detections[0]
                    is_known_circle = True
                    detected_circles[i] = (x, y, 0, detections)
                    break

            if not is_known_circle:
                detected_circles.append((x, y, 0, [(x, y, r)]))
                matched_circles.append(len(detected_circles) - 1)

        print("matched", matched_circles)

        circles_to_remove = list()
        for (x, y, misses, detections) in detected_circles:
            i = detected_circles.index((x, y, misses, detections))

            detected = False
            for idx in matched_circles:
                detected = detected or idx == i

            if not detected:
                detected_circles[i] = (x, y, misses + 1, detections)
                if misses > DETECTION_MISSES_TILL_REMOVED:
                    circles_to_remove.append(i)

        for idx in reversed(circles_to_remove):
            del detected_circles[idx]

        for (x, y, misses, detections) in detected_circles:
            if len(detections) >= MIN_DETECTIONS_TILL_PERCEIVED_AS_BOTTLE:

                pos_x = 0
                pos_y = 0
                radius = 0

                for (x_c, y_c, r) in detections:
                    pos_x += x_c
                    pos_y += y_c
                    radius += r

                pos_x /= len(detections)
                pos_y /= len(detections)
                radius /= len(detections)

                num_bottles_top += 1
                cv2.circle(top_image, (np.round(pos_x).astype("int"), np.round(pos_y).astype("int")),
                           np.round(radius).astype("int"), (0, 0, 255), 4)

    return num_bottles_top, top_image


def main():
    camera_capture = CameraCapture()

    detected_circles = list()

    notification_sent = False

    while True:
        complete_image = camera_capture.get_current_image_rotated()
        beerfridge_image = get_beerfridge_image(complete_image=complete_image)
        gray_beer = grayscale_image(image=beerfridge_image)

        blurred_beer = cv2.GaussianBlur(gray_beer, (9, 9), 10)
        blurred_beer = cv2.addWeighted(gray_beer, 1.5, blurred_beer, -0.5, 0, gray_beer)

        blurred_beer2 = cv2.GaussianBlur(blurred_beer, (5, 5), 10)
        blurred_beer2 = cv2.addWeighted(blurred_beer, 1.5, blurred_beer2, -0.5, 0, blurred_beer)

        blurred_beer = threshold_image(image=blurred_beer2)

        top_fridge = blurred_beer[0:360, 0:460]
        bottom_fridge = beerfridge_image[360:610, 0:460]

        num_bottles_top, top_image_detected = detect_bottles_with_hough_transform(detected_circles, top_fridge, beerfridge_image.copy())

        num_bottles_bottom, bottom_image_detected = detect_bottles_with_ssd(bottom_fridge)
        top_image_detected[360:610, 0:460] = bottom_image_detected

        total_bottles = num_bottles_top + num_bottles_bottom
        print("total bottles in picture: ", total_bottles)

        if total_bottles <= MESSAGE_SENDING_THRESHOLD_BOTTLE_COUNT:
            current_date = datetime.datetime.fromtimestamp(time.time(), tz=datetime.timezone.utc).isoformat()

            if not notification_sent:
                send_rabbit_message(fridge_id="bit_beer_fridge",
                                    date=current_date,
                                    beers_left_in_fridge=total_bottles,
                                    beer_threshold=MESSAGE_SENDING_THRESHOLD_BOTTLE_COUNT)
                notification_sent = True

        # show the output image
        display_image("Detected bottles", top_image_detected)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera_capture.stop()
    cv2.destroyAllWindows()


def get_beerfridge_image(complete_image):
    return get_region_of_interest(image=complete_image,
                                  x=REGION_OF_INTEREST["x"],
                                  y=REGION_OF_INTEREST["y"],
                                  w=REGION_OF_INTEREST["width"],
                                  h=REGION_OF_INTEREST["height"])


def get_region_of_interest(image, x, y, w, h):
    return image[y:(y + h), x:(x + w)]


def display_image(window_name, image):
    cv2.imshow(winname=window_name, mat=image)


def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold_image(image):
    success, threshold = cv2.threshold(src=image, thresh=40, maxval=100, type=cv2.THRESH_TOZERO)
    equalized = cv2.equalizeHist(threshold)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(equalized)


def is_within(value, lower_bound, upper_bound):
    return value >= lower_bound and value <= upper_bound


def send_rabbit_message(fridge_id, date, beers_left_in_fridge, beer_threshold):
    rabbit_msg_string = create_rabbit_message_string(fridge_id=fridge_id, date=date,
                                                     beers_left_in_fridge=beers_left_in_fridge,
                                                     beer_threshold=beer_threshold)

    credentials = pika.credentials.PlainCredentials(username="username", password="password")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost", credentials=credentials))
    channel = connection.channel()

    channel.basic_publish(exchange="x.event_hansi_beerfridge_threshold_exceeded.internal",
                          routing_key="hello",
                          body=rabbit_msg_string)


def create_rabbit_message_string(fridge_id, date, beers_left_in_fridge, beer_threshold):
    main_payload = {
        "fridgeId": fridge_id,
        "date": date,
        "beersLeftInFridge": beers_left_in_fridge,
        "beerThreshold": beer_threshold
    }

    main_payload_json = json.dumps(main_payload)

    container_message = {
        "type": "EVENT_HANSI_BEERFRIDGE_THRESHOLD_EXCEEDED",
        "payload": main_payload_json,
        "metaData": {
            "uuid": str(uuid.uuid4())
        }
    }

    container_message_json = json.dumps(container_message)

    return container_message_json


if __name__ == "__main__":
    main()
