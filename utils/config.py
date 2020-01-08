import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# -------------
import configparser
import numpy as np
import logging

logger = logging.getLogger(__name__)
CONFIG_FILE_PATH = './setting.ini'


class configInit():
  def __init__(self):
    config_ini = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE_PATH):
      logger.error("設定ファイルがありません")
      exit(-1)
    config_ini.read(CONFIG_FILE_PATH, encoding='utf-8')
    logging.info("----設定開始----")
    try:
      self.CamID = int(config_ini["DEFAULT"]["CameraID"])
      
      self.PlotOK = int(config_ini["DEFAULT"]["Plot"])
      if self.PlotOK:
        self.PlotOK = True
      else:
        self.PlotOK = False

      self.CamWidth = int(config_ini["DEFAULT"]["CamWIDTH"]) 
      self.CamHeight = int(config_ini["DEFAULT"]["CamHEIGHT"]) 
      self.Record = int(config_ini["DEFAULT"]["Record"])
      if self.Record == 1:
        self.Record = True
      else:
        self.Record = False

      self.TrakingFailureTh = int(config_ini["PARAM"]["TrackingFailureTh"])
      self.FaceLandmarkFailureTh = int(config_ini["PARAM"]["FaceLandmarkDetectFailureTh"])
      self.MoveOffset = int(config_ini["PARAM"]["MoveOffset"])
      self.RotXRange = float(config_ini["PARAM"]["RotXRange"]) * np.pi / 180.
      self.RotYRange = float(config_ini["PARAM"]["RotYRange"]) * np.pi / 180.
      self.RotZRange = float(config_ini["PARAM"]["RotZRange"]) * np.pi / 180.
      

    except Exception as e:
      logger.critical(e)
      exit(-1)



    logger.info("----設定完了----")

