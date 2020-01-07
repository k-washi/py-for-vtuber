import numpy as np

class Postionlkf():
  def __init__(self, init_w, init_h):
    self.T = 1.
    #state space model x = A*x + B*u + w, w ~ N(0, Q)
    self.A = np.array([[1, 0, self.T, 0], [0, 1, 0, self.T], [0, 0, 1, 0], [0, 0, 0, 1]])
    #self.B = np.array([[1,0], [0, 1]])
    self.Q = np.array([[100, 0, 0, 0],[0, 100, 0, 0], [0, 0, 500, 0], [0, 0, 0, 500]]) #移動誤差

    #observe model y = C * x + v, v ~ N(0, R)
    self.C = np.array([[1,0, 0, 0], [0,1, 0, 0]])
    self.R = np.array([[200, 0],[0, 200]]) #計測誤差

    self.x = np.array([[init_w], [init_h], [0], [0]])

    self.mu = np.array([[0], [0], [0], [0]])
    self.Sigma = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])


  def update(self, w, h, update = True):
    #estiarrayion
    
    mu_ = np.dot(self.A, self.x) #+u速度の内界情報なし #現在の推定値
    #print("mu_ {0}".format(mu_))
    Sigma_ = self.Q + np.dot(np.dot(self.A, self.Sigma), self.A.T) #現在の誤差行列

    #update
    if update:
      yi = np.array([[w], [h]]) - np.dot(self.C, mu_) #観測残差
      #print("yi {0}".format(yi))
      S = np.dot(np.dot(self.C, Sigma_), self.C.T) + self.R #観測残差の共分散
      K = np.dot(np.dot(Sigma_, self.C.T), np.linalg.inv(S))  #カルマンゲイン
      self.x = mu_ + np.dot(K, yi)#更新された現在の推定値
      self.Sigma = Sigma_ - np.dot(np.dot(K, self.C), Sigma_)#更新された現在の誤差行列
      print("x {0}".format(self.x))
    
      return np.array([self.x[0][0], self.x[1][0]]), self.Sigma
    
    return np.array([mu_[0][0], mu_[0][1]]), Sigma_

class Rotationlkf():
  def __init__(self, init_x, init_y, init_z):
    
    self.T = 1.
    #state space model x = A*x + B*u + w, w ~ N(0, Q)
    self.A = np.array([[1, 0, self.T, 0], [0, 1, 0, self.T], [0, 0, 1, 0], [0, 0, 0, 1]])
    #self.B = np.array([[1,0], [0, 1]])
    self.Q = np.array([[100, 0, 0, 0],[0, 100, 0, 0], [0, 0, 500, 0], [0, 0, 0, 500]]) #移動誤差

    #observe model y = C * x + v, v ~ N(0, R)
    self.C = np.array([[1,0, 0, 0], [0,1, 0, 0]])
    self.R = np.array([[200, 0],[0, 200]]) #計測誤差

    self.x = np.array([[init_x], [init_y], [0], [0]])

    self.mu = np.array([[0], [0], [0], [0]])
    self.Sigma = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])


  def update(self, w, h, update = True):
    #estiarrayion
    
    mu_ = np.dot(self.A, self.x) #+u速度の内界情報なし #現在の推定値
    #print("mu_ {0}".format(mu_))
    Sigma_ = self.Q + np.dot(np.dot(self.A, self.Sigma), self.A.T) #現在の誤差行列

    #update
    if update:
      yi = np.array([[w], [h]]) - np.dot(self.C, mu_) #観測残差
      #print("yi {0}".format(yi))
      S = np.dot(np.dot(self.C, Sigma_), self.C.T) + self.R #観測残差の共分散
      K = np.dot(np.dot(Sigma_, self.C.T), np.linalg.inv(S))  #カルマンゲイン
      self.x = mu_ + np.dot(K, yi)#更新された現在の推定値
      self.Sigma = Sigma_ - np.dot(np.dot(K, self.C), Sigma_)#更新された現在の誤差行列
      print("x {0}".format(self.x))
    
      return np.array([self.x[0][0], self.x[1][0]]), self.Sigma
    
    return np.array([mu_[0][0], mu_[0][1]]), Sigma_