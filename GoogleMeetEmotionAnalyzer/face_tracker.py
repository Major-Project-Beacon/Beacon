import math


class FaceTracker:

    def __init__(self, max_distance=100):

        self.next_id = 1

        self.faces = {}

        self.max_distance = max_distance

    def center(self, box):

        x, y, w, h = box

        return (
            x + w // 2,
            y + h // 2
        )

    def distance(self, point1, point2):

        return math.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )

    def update(self, detected_boxes):

        new_faces = {}

        used_ids = set()

        for box in detected_boxes:

            current_center = self.center(box)

            best_id = None

            best_distance = self.max_distance

            for face_id, old_data in self.faces.items():

                if face_id in used_ids:
                    continue

                old_center = old_data["center"]

                distance = self.distance(
                    current_center,
                    old_center
                )

                if distance < best_distance:

                    best_distance = distance

                    best_id = face_id

            if best_id is None:

                best_id = self.next_id

                self.next_id += 1

            new_faces[best_id] = {
                "box": box,
                "center": current_center
            }

            used_ids.add(best_id)

        self.faces = new_faces

        return new_faces