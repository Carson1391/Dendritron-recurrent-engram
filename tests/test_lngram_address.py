from __future__ import annotations

import unittest

import numpy as np

from dendritron.lngram_address import (
    address_sequence,
    lngram_addresses,
    pack_route_bits,
    table_rows,
)


class LNGramAddressTests(unittest.TestCase):
    def test_pack_four_bits(self):
        bits = np.array([[[0, 1, 0, 1], [1, 1, 1, 1]]])
        np.testing.assert_array_equal(pack_route_bits(bits), [[10, 15]])

    def test_route_partitioned_bigram_addresses(self):
        symbols = np.array([[[1, 2], [3, 4], [5, 6]]])
        addresses, valid = lngram_addresses(
            symbols,
            order=2,
            alphabet_size=16,
        )
        self.assertFalse(valid[0, 0].any())
        self.assertEqual(
            int(addresses[0, 1, 0]),
            address_sequence([1, 3], route=0, alphabet_size=16),
        )
        self.assertEqual(
            int(addresses[0, 1, 1]),
            address_sequence([2, 4], route=1, alphabet_size=16),
        )
        self.assertNotEqual(addresses[0, 1, 0], addresses[0, 1, 1])

    def test_expected_table_sizes(self):
        self.assertEqual(table_rows(512, 16, 2), 131_072)
        self.assertEqual(table_rows(512, 16, 3), 2_097_152)


if __name__ == "__main__":
    unittest.main()
