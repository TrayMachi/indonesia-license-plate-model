import unittest

import numpy as np

from src.postprocess import decode_yolo11_output


class DecodeYolo11OutputTest(unittest.TestCase):
    def test_transposes_filters_and_suppresses_overlapping_boxes(self):
        output = np.array(
            [
                [100.0, 101.0, 300.0],
                [100.0, 101.0, 300.0],
                [40.0, 40.0, 30.0],
                [40.0, 40.0, 30.0],
                [0.90, 0.80, 0.10],
            ],
            dtype=np.float32,
        )[None, ...]

        boxes, scores, class_ids = decode_yolo11_output(output)

        self.assertEqual(boxes.shape, (1, 4))
        np.testing.assert_allclose(boxes[0], (80.0, 80.0, 120.0, 120.0))
        np.testing.assert_allclose(scores, (0.90,))
        np.testing.assert_array_equal(class_ids, (0,))

    def test_accepts_transposed_prediction_layout(self):
        output = np.array(
            [[[80.0, 80.0, 40.0, 40.0, 0.95], [200.0, 200.0, 20.0, 20.0, 0.85]]],
            dtype=np.float32,
        )

        boxes, scores, class_ids = decode_yolo11_output(output)

        self.assertEqual(boxes.shape, (2, 4))
        np.testing.assert_allclose(boxes[0], (60.0, 60.0, 100.0, 100.0))
        np.testing.assert_allclose(scores, (0.95, 0.85))
        np.testing.assert_array_equal(class_ids, (0, 0))

    def test_scales_normalized_coordinates_to_model_pixels(self):
        output = np.array(
            [[[0.5], [0.5], [0.25], [0.25], [0.95]]],
            dtype=np.float32,
        )

        boxes, scores, class_ids = decode_yolo11_output(output, input_size=(640, 640))

        np.testing.assert_allclose(boxes, ((240.0, 240.0, 400.0, 400.0),))
        np.testing.assert_allclose(scores, (0.95,))
        np.testing.assert_array_equal(class_ids, (0,))


if __name__ == "__main__":
    unittest.main()
