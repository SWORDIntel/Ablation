import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
from framewerx.aegis_lab.hardware.thermal import ThermalGuardian, ThermalLevel

class TestThermalUnit(unittest.TestCase):
    @patch('os.path.exists')
    @patch('os.listdir')
    def setUp(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["thermal_zone0", "thermal_zone1"]
        self.tg = ThermalGuardian()

    @patch('builtins.open', new_callable=mock_open, read_data="45000")
    def test_get_max_temperature(self, mock_file):
        temp = self.tg.get_max_temperature()
        self.assertEqual(temp, 45.0)

    @patch('builtins.open', new_callable=mock_open, read_data="80000")
    def test_thermal_level_high(self, mock_file):
        level = self.tg.get_thermal_level()
        self.assertEqual(level, ThermalLevel.HIGH)

    @patch('builtins.open', new_callable=mock_open, read_data="95000")
    def test_thermal_status_critical(self, mock_file):
        status = self.tg.get_status()
        self.assertEqual(status["level"], "CRITICAL")
        self.assertFalse(status["safe_to_compute"])
        self.assertTrue(status["throttling_recommended"])

if __name__ == "__main__":
    unittest.main()
