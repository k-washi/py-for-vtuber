import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#----------
import cv2

#----------

from utils import echeck


class readIMG():
  def __init__(self):
    pass
    self.cimg = None
  
  def ReadCIMG(self, filepath):
    try:
      self.cimg = cv2.imread(filepath) #1: color, 0: gray
      if self.cimg is None:
        raise FileExistsError("Can not read img " + str(file_path))
    except FileExistsError as e:
      echeck.error_print(e)
      exit(-1)

   
  
  def ShowIMG(self):
    from matplotlib import pyplot as plt
    img = cv2.cvtColor(self.cimg, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
  file_path = './testdata/lena.jpeg'
  img = readIMG()
  img.ReadCIMG(file_path)
  img.ShowIMG()
