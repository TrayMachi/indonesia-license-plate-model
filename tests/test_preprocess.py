import unittest

import numpy as np

from src.preprocess import to_model_input


class ToModelInputTest(unittest.TestCase):
    def test_returns_rgb_float32_nchw_tensor(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[0, 0] = (0, 128, 255)

        tensor = to_model_input(image)

        self.assertEqual(tensor.shape, (1, 3, 2, 3))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], (0, 128, 255), atol=1 / 255)
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
