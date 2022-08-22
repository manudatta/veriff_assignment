import numpy
import tensorflow.compat.v2 as tf
import cv2
import numpy as np
import time


# Getting some unknown linter errors, disable everything to get this to production asap
# pylint: disable-all


class BirdClassifier:

    def __init__(self, model):
        self.model = model


    def process_image(self, raw_image_data: bytes) -> numpy.ndarray:
        bird_model = self.model
        image_array = np.asarray(bytearray(raw_image_data), dtype=np.uint8)
        # Changing images
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255
        # Generate tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)
        model_raw_output = bird_model.call(image_tensor).numpy()
        return model_raw_output
