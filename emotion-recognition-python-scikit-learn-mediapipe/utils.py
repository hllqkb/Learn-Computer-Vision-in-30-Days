import cv2
import mediapipe as mp
def get_face_landmarks(image, draw=False, static_image_mode=True):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_landmarks = []

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_input_rgb)

        if not results.multi_face_landmarks:
            return []

        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        coords = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in results.multi_face_landmarks[0].landmark
        ]
        min_x, min_y, min_z = (
            min(c[i] for c in coords) for i in range(3)
        )
        image_landmarks = [
            coord - min_val
            for x, y, z in coords
            for coord, min_val in [(x, min_x), (y, min_y), (z, min_z)]
        ]

    return image_landmarks
