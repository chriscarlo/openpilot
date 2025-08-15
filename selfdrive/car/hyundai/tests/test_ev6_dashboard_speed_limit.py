#!/usr/bin/env python3
"""
TDD Tests for EV6 Dashboard Speed Limit functionality
These tests verify that dashboard speed limit is correctly parsed from ECAN bus for CANFD cars
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from parameterized import parameterized

from cereal import car
from opendbc.car.hyundai.values import CAR, HyundaiFlags, CANFD_CAR
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.carstate import CarState


class TestEV6DashboardSpeedLimit(unittest.TestCase):
    """Test suite for EV6 dashboard speed limit functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.CP = Mock()
        self.CP.carFingerprint = CAR.KIA_EV6
        self.CP.flags = 0
        
    def test_ev6_is_canfd_car(self):
        """Test that EV6 is recognized as a CANFD car"""
        self.assertIn(CAR.KIA_EV6, CANFD_CAR)
        
    def test_fingerprint_check_uses_ecan_for_canfd(self):
        """Test that CANFD cars check ECAN bus for 0x1FA, not CAM bus"""
        # Create mock fingerprint with 0x1FA on ECAN bus
        fingerprint = {
            0: {},  # PT bus
            1: {},  # CAM bus  
            2: {0x1FA: 8}  # ECAN bus - 0x1FA present here!
        }
        
        candidate = CAR.KIA_EV6
        
        # This should check ECAN (bus 2) for CANFD cars
        with patch('opendbc.car.hyundai.values.get_platform_codes', return_value=set()):
            ret = CarInterface.get_params(candidate, fingerprint, car_fw=[], alpha_long=False, is_release=False, docs=False)
            
        # For CANFD cars, 0x1FA on ECAN should set the dashboard speed limit flag
        self.assertTrue(ret.flags & HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR, 
                       "Dashboard speed limit flag should be set when 0x1FA is on ECAN for CANFD cars")
    
    def test_fingerprint_cam_bus_does_not_set_flag_for_canfd(self):
        """Test that 0x1FA on CAM bus does NOT set flag for CANFD cars"""
        # Create mock fingerprint with 0x1FA on CAM bus (wrong bus!)
        fingerprint = {
            0: {},  # PT bus
            1: {0x1FA: 8},  # CAM bus - 0x1FA here (wrong for CANFD!)
            2: {}  # ECAN bus - no 0x1FA
        }
        
        candidate = CAR.KIA_EV6
        
        with patch('opendbc.car.hyundai.values.get_platform_codes', return_value=set()):
            ret = CarInterface.get_params(candidate, fingerprint, car_docs=[], car_fw=[], experimental_long=False, docs=False)
            
        # Should NOT set flag because 0x1FA is on wrong bus for CANFD
        self.assertFalse(ret.flags & HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR,
                        "Dashboard speed limit flag should NOT be set when 0x1FA is only on CAM bus for CANFD cars")
    
    def test_carstate_parses_fr_cmr_message_when_flag_set(self):
        """Test that CarState adds FR_CMR_02_100ms to parser when flag is set"""
        CP = Mock()
        CP.carFingerprint = CAR.KIA_EV6
        CP.flags = HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR
        CP.openpilotLongitudinalControl = False
        CP.enableBsm = False
        
        # Check that the message is added to the parser
        messages = CarState.get_can_parser(CP)
        
        # The parser should include FR_CMR_02_100ms message
        message_names = [msg[0] for msg in messages._messages if hasattr(messages, '_messages')]
        self.assertIn("FR_CMR_02_100ms", message_names, 
                     "FR_CMR_02_100ms should be in CAN parser when flag is set")
    
    def test_calculate_speed_limit_for_canfd(self):
        """Test the calculate_speed_limit method for CANFD cars"""
        cs = CarState(Mock())
        cs.CP = Mock()
        cs.CP.carFingerprint = CAR.KIA_EV6
        cs.CP.flags = HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR
        
        # Mock the CAN parser values
        cp = Mock()
        cp_cam = Mock()
        
        # For CANFD with dashboard speed limit, it should read from FR_CMR_02_100ms
        cp.vl = {
            "FR_CMR_02_100ms": {
                "ISLW_SpdCluMainDis": 80  # 80 km/h speed limit
            }
        }
        
        # The method should extract the speed limit correctly
        speed_limit = cs.calculate_speed_limit(cp, cp_cam)
        self.assertEqual(speed_limit, 80, "Should correctly parse speed limit from FR_CMR_02_100ms")
    
    def test_speed_limit_flows_to_ret_sp(self):
        """Test that parsed speed limit flows to ret_sp.speedLimit"""
        cs = CarState(Mock())
        cs.CP = Mock()
        cs.CP.carFingerprint = CAR.KIA_EV6
        cs.CP.flags = HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR
        cs.CP.openpilotLongitudinalControl = False
        cs.CP.enableBsm = False
        
        # Mock CAN data with speed limit
        cp = Mock()
        cp_cam = Mock()
        cp.vl = {
            "CLU11": {"CF_Clu_SPEED_UNIT": 0},  # Metric
            "FR_CMR_02_100ms": {
                "ISLW_SpdCluMainDis": 60  # 60 km/h
            },
            # Add other required messages for update
            "MDPS12": {"CF_Mdps_StrColTq": 0, "CF_Mdps_FailStat": 0, "CF_Mdps_MsgCount2": 0,
                      "CF_Mdps_SErr": 0, "CR_Mdps_StrTq": 0, "CF_Mdps_CurrErr": 0,
                      "CF_Mdps_Chksum2": 0},
            "TCS13": {"DriverOverride": 0, "ESC_Off": 0, "TQI": 0, "TQI_2": 0, "BrakeLight": 0},
            "TCS11": {"TCS_PAS": 0, "ABS_W_LAMP": 0},
            "EMS11": {"PV_AV_CAN": 40},
            "WHL_SPD11": {"WHL_SPD_FL": 0, "WHL_SPD_FR": 0, "WHL_SPD_RL": 0, "WHL_SPD_RR": 0},
            "SAS11": {"SAS_Angle": 0, "SAS_Speed": 0},
            "SCC11": {"MainMode_ACC": 0, "VSetDis": 0, "AliveCounterACC": 0, "ACC_ObjStatus": 0, 
                     "ACC_ObjDist": 0, "ACC_ObjRelSpd": 0, "TauGapSet": 0},
            "SCC12": {"CF_VSM_Prefill": 0, "CF_VSM_DecCmdAct": 0, "CF_VSM_HBACmd": 0,
                     "CF_VSM_Warn": 0, "CF_VSM_Stat": 0, "CF_VSM_BeltCmd": 0, "ACCMode": 0,
                     "StopReq": 0, "CR_VSM_DecCmd": 0, "aReqRaw": 0, "TakeOverReq": 0,
                     "PreFill": 0, "aReqValue": 0, "CF_VSM_ConfMode": 0, "AEB_Failinfo": 0,
                     "AEB_Status": 0, "AEB_CmdAct": 0, "AEB_StopReq": 0, "CR_VSM_Alive": 0,
                     "CR_VSM_ChkSum": 0},
            "SCC14": {"HDA_Icon": 0}
        }
        
        # Configure other required CP attributes
        cs.is_metric = True
        
        # Call update to process the speed limit
        ret, ret_sp = cs.update(cp, cp_cam, Mock())
        
        # Speed limit should be set in ret_sp
        self.assertEqual(ret_sp.speedLimit, 60.0 * 0.277778,  # Convert km/h to m/s
                        "Speed limit should flow to ret_sp.speedLimit")
    
    @parameterized.expand([
        (0, 0.0),      # 0 km/h
        (30, 8.333),   # 30 km/h ≈ 8.33 m/s
        (60, 16.667),  # 60 km/h ≈ 16.67 m/s
        (100, 27.778), # 100 km/h ≈ 27.78 m/s
        (255, 0.0),    # 255 = invalid/no limit
    ])
    def test_speed_limit_conversion(self, input_kmh, expected_ms):
        """Test speed limit conversion from km/h to m/s"""
        cs = CarState(Mock())
        cs.CP = Mock()
        cs.CP.carFingerprint = CAR.KIA_EV6
        cs.CP.flags = HyundaiFlags.HAS_DASHBOARD_SPEED_LIMIT_FR_CMR
        
        # Mock CAN data
        cp = Mock()
        cp_cam = Mock()
        cp.vl = {
            "FR_CMR_02_100ms": {
                "ISLW_SpdCluMainDis": input_kmh
            }
        }
        
        speed_limit = cs.calculate_speed_limit(cp, cp_cam)
        
        # Convert to m/s for comparison
        if input_kmh not in (0, 255):
            speed_limit_ms = speed_limit * 0.277778  # km/h to m/s
            self.assertAlmostEqual(speed_limit_ms, expected_ms, places=2,
                                 msg=f"Speed limit {input_kmh} km/h should convert to {expected_ms} m/s")
        else:
            # Invalid values should return 0
            self.assertEqual(speed_limit, 0.0, f"Invalid value {input_kmh} should return 0")


if __name__ == "__main__":
    unittest.main()