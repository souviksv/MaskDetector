# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np


class FaceDetector:
    def __init__(self, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        model_path = 'models/model.pb'
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def detect_face(self, image, score_threshold=0.5):
        """Detect faces.
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].
        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        image_cp = image
        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        if boxes is None:
            return None

        for box in boxes:
            ymin, xmin, ymax, xmax = box
        
            h = ymax - ymin
            w = xmax - xmin

            im_w, im_h = image_cp.shape[:2]

            ymin = int(max(ymin - (w * 0.35), 0))
            ymax = int(min(ymax + (w * 0.35), im_w))

            xmin = int(max(xmin - (h * 0.35), 0))
            xmax = int(min(xmax + (h * 0.35), im_h))

            cropped_face = image_cp[ymin:ymax, xmin:xmax]
            
            # cv2.imshow('CroppedImage', cropped_face)
            # cv2.waitKey(1)

            return cropped_face