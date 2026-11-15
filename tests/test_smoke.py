"""Smoke tests: model builds, split/merge round trip, PSNR, shared constants."""
import math

import numpy as np

from common.header import DATA_ROOT
from common.utils import merge_image2, psnr, split2


def test_generator_builds_and_predicts():
    from service.Generator import Generator

    model = Generator(biggest_layer=16)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 256, 256, 1)
    out = model.predict(np.zeros((1, 256, 256, 1), dtype=np.float32), verbose=0)
    assert out.shape == (1, 256, 256, 1)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0  # sigmoid output


def test_split_merge_round_trip():
    h, w = 512, 768
    image = np.random.default_rng(0).random((1, h, w, 1)).astype(np.float32)
    tiles = split2(image, 1, h, w)
    assert tiles.shape == ((h // 256) * (w // 256), 256, 256, 1)
    merged = merge_image2(tiles, h, w)
    assert merged.shape == (h, w, 1)
    np.testing.assert_array_equal(merged, image[0])


def test_psnr_identical_images_is_100():
    img = np.full((8, 8), 0.5, dtype=np.float32)
    assert psnr(img, img) == 100


def test_psnr_known_value():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 0.1, dtype=np.float32)
    # MSE = 0.01 -> 20 * log10(1 / 0.1) = 20 dB
    assert math.isclose(psnr(a, b), 20.0, rel_tol=1e-4)


def test_data_root_is_shared_constant():
    assert DATA_ROOT == "./data_wm"
