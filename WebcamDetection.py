import mediapipe as mp
import cv2
import time


class HolisticDetector :
    def __init__(self, static_mode = False, model_complexity = 1, smooth_landmarks = True,
                 min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5) :
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(self.static_mode, self.model_complexity, self.smooth_landmarks,
                                                  self.min_detection_confidence, self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness = 1, circle_radius = 2)

    def find_mesh(self, img, draw = True) :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False

        results = self.holistic.process(img)

        img.flags.writeable = True

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if draw :
            # Face
            self.mp_draw.draw_landmarks(img, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS,
                                        self.mp_draw.DrawingSpec(color = (80, 110, 10), thickness = 1,
                                                                 circle_radius = 1),
                                        self.mp_draw.DrawingSpec(color = (80, 256, 121), thickness = 1,
                                                                 circle_radius = 1))

            # Left Hand
            self.mp_draw.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_draw.DrawingSpec(color = (80, 22, 10), thickness = 2,
                                                                 circle_radius = 4),
                                        self.mp_draw.DrawingSpec(color = (80, 44, 121), thickness = 2,
                                                                 circle_radius = 2))

            # Right Hand
            self.mp_draw.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                        self.mp_draw.DrawingSpec(color = (121, 22, 76), thickness = 2,
                                                                 circle_radius = 4),
                                        self.mp_draw.DrawingSpec(color = (121, 44, 250), thickness = 2,
                                                                 circle_radius = 2))

            # Pose Detections
            self.mp_draw.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                        self.mp_draw.DrawingSpec(color = (245, 117, 66), thickness = 2,
                                                                 circle_radius = 4),
                                        self.mp_draw.DrawingSpec(color = (245, 66, 230), thickness = 2,
                                                                 circle_radius = 2))

        return img


class HandsDetector :
    def __init__(self, static_mode = False, max_hands = 1, min_detection_confidence = 0.5,
                 min_tracking_confidence = 0.5) :
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_mode, self.max_hands, self.min_detection_confidence,
                                         self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness = 1, circle_radius = 2)

    def find_mesh(self, img, draw = True) :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = self.hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks :
            for num, hand in enumerate(results.multi_hand_landmarks) :
                if draw :
                    self.mp_draw.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(color = (121, 22, 76), thickness = 2,
                                                                         circle_radius = 4),
                                                self.mp_draw.DrawingSpec(color = (250, 44, 250), thickness = 2,
                                                                         circle_radius = 2))

        return image


class FaceDetector :
    def __init__(self, static_mode = False, max_faces = 2, min_detection_confidence = 0.5,
                 max_tracking_confidence = 0.5) :
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.max_tracking_confidence = max_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_face_mesh.FaceMesh(self.static_mode, self.max_faces,
                                                self.min_detection_confidence, self.max_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness = 1, circle_radius = 2)

    def find_mesh(self, img, draw = True) :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = self.hands.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks :
            for facial_landmarks in results.multi_face_landmarks :
                if draw :
                    self.mp_draw.draw_landmarks(image, facial_landmarks, self.mp_face_mesh.FACE_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(color = (121, 22, 76), thickness = 1,
                                                                         circle_radius = 1),
                                                self.mp_draw.DrawingSpec(color = (250, 44, 250), thickness = 1,
                                                                         circle_radius = 1))

        return image


def main() :
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    p_time = 0

    detector = FaceDetector()

    while video_capture.isOpened() :
        success, frame = video_capture.read()

        frame = cv2.flip(frame, 1)

        frame = detector.find_mesh(frame)

        if not success :
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS : {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Mediapipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()
