from numpy import array, pi, sqrt, arange, \
                exp, matrix, zeros, concatenate, linspace, diag
from pylab import plot, legend
from numpy.random import normal

class MonteCarlo:
    def __init__(self,record_trajectory=True):
        self.shape_pdc = None
        self.shape_cool = None
        self.record_trajectory = record_trajectory

    def define_dynamics(self, 
                        kappa=2*pi*360e3,
                        kappa_e=2*pi*180e3,
                        gamma_b=2*pi*17e3,
                        gamma_r=2*pi*11e3,
                        gamma_m=2*pi*35):
        self.kappa_e = kappa_e
        self.gamma_b = gamma_b
        self.gamma_r = gamma_r
        
        self.M = array([[-(kappa/2), sqrt(gamma_b*kappa)/2, 0, sqrt(gamma_r*kappa)/2],
               [sqrt(gamma_b*kappa)/2, -(gamma_m/2), sqrt(gamma_r*kappa)/2, 0],
               [0,-(sqrt(gamma_r*kappa)/2),-(kappa/2),-(sqrt(gamma_b*kappa)/2)],
               [-sqrt(gamma_r*kappa)/2,0,-sqrt(gamma_b*kappa)/2,-gamma_m/2]])
        """
        
        self.M = array([[-(kappa/2), 0, 0, -sqrt(gamma_b*kappa)/2 + sqrt(gamma_r*kappa)/2],
               [0, -(gamma_m/2),-sqrt(gamma_b*kappa)/2 + sqrt(gamma_r*kappa)/2, 0],
               [0,-(sqrt(gamma_r*kappa)/2)-(sqrt(gamma_b*kappa)/2),-(kappa/2),0],
               [-sqrt(gamma_r*kappa)/2-sqrt(gamma_b*kappa)/2,0,0,-gamma_m/2]])
        """   
        self.K = array([[sqrt(kappa_e), sqrt(kappa - kappa_e), 0, 0, 0, 0], 
             [0, 0, sqrt(gamma_m), 0, 0, 0,],
             [0, 0, 0, sqrt(kappa_e), sqrt(kappa - kappa_e), 0],
             [0, 0, 0, 0, 0, sqrt(gamma_m)]])
        
    def step(self, dt):
        noise = sqrt(dt)*normal(size=6)
        #noise[2]*=4
        #noise[5]*=4
        self.vec = self.vec + dt*self.M.dot(self.vec) + self.K.dot(noise)
        self.t = self.t+dt
        if self.record_trajectory:
            self.times.append(self.t)
            self.vecs.append(self.vec)
        return array([sqrt(self.kappa_e)*self.vec[0] - noise[0]/dt, sqrt(self.kappa_e)*self.vec[2]- noise[3]/dt])
    
    def shape(self, gamma, t_max, dt, sign=+1):
        """norm = 2*dt*sqrt(gamma/(1-exp(-gamma*t_max)))
        if sign==-1:
            ser = arange(0, t_max, dt)
        else:
            ser = t_max - arange(0, t_max, dt)
        return norm*exp(-gamma*ser/2) 
        """
        shape_cool = exp(sign*gamma*arange(0, t_max, dt)/2)
        return sqrt(dt)*shape_cool/sqrt((shape_cool**2).sum())
        
    
    def run(self, t_max, dt, gamma_r=2*pi*11e3, gamma_b=2*pi*17e3, calibrate=False):
        self.vec = array([0,0,0,0])
        self.times = [0]
        self.vecs = [self.vec]
        self.t = 0
        self.pulse_pdc = []
        
        ### Thermalize resonator
        self.define_dynamics(gamma_r=0, gamma_b=0, gamma_m=2*pi*30e3)
        for t in arange(0, t_max, dt):
            self.step(dt)
        
        
        self.define_dynamics(gamma_r=0, gamma_b=gamma_b*(not calibrate))
        
        if self.shape_pdc is None:
            self.shape_pdc = self.shape(gamma_b, t_max, dt)
        
        for t in arange(0, t_max, dt):
            self.pulse_pdc.append(self.step(dt))
        self.pulse_pdc = array(self.pulse_pdc)
        
        #self.vec[0] = 0
        #self.vec[2] = 0
        
        self.define_dynamics(gamma_b=0,gamma_r=gamma_r*(not calibrate))
        self.pulse_cool = []
        t_max_cool = t_max
        if self.shape_cool is None:
            self.shape_cool = self.shape(gamma_r, t_max_cool, dt, sign=-1)
        
        for t in arange(0, t_max_cool, dt):
            self.pulse_cool.append(self.step(dt))
            
        self.pulse_cool = array(self.pulse_cool)
        
        self.quad_pdc, self.quad_cool = self.get_quads(t_max, dt)
        
    def get_quads(self, t_max, dt):
        return (self.shape_pdc.dot(self.pulse_pdc),
                self.shape_cool.dot(self.pulse_cool))
        
    def run_stats(self, t_max, dt, n, gamma_b=2*pi*17e3, gamma_r=2*pi*11e3):
        ## calibration shot noise first
        """cov_total = matrix(zeros((4, 4)))
        for i in range(n):       
            if i%10==0:
                print "calibration trajectory " + str(i)
            self.run(t_max, dt, gamma_b=gamma_b, gamma_r=gamma_r, calibrate=True)
            vect = concatenate((self.quad_pdc, self.quad_cool))
            cov = matrix(vect).transpose()*matrix(vect)
            cov_total += cov 
        cov_cal = cov_total/n
        self.cov_cal = diag(1/array(cov_cal.diagonal())[0])
#        self.vacuum = self.cov_cal.trace()/4    
        """
        ### real run
        self.quads_pdc = []
        self.quads_cool = []
        cov_total = matrix(zeros((4, 4)))
        for i in range(n):
            if i%10==0:
                print "trajectory " + str(i)
            self.run(t_max, dt, gamma_b=gamma_b, gamma_r=gamma_r)
            self.quads_pdc.append(self.quad_pdc)
            self.quads_cool.append(self.quad_cool)
            vect = concatenate((self.quad_pdc, self.quad_cool))
            cov = matrix(vect).transpose()*matrix(vect)
            cov_total+= cov
        self.quads_pdc  = array(self.quads_pdc)
        self.quads_cool = array(self.quads_cool)
        self.cov = cov_total/n
        """self.cov = self.cov*self.cov_cal
        """
    def plot_last_trajectory(self):
        plot(self.vecs)
        legend(["Xres", "Xm", "Yres", "Ym"])
        
    def plot_last_pulses(self):
        plot(self.pulse_pdc)
        plot(self.shape_pdc*20000)
        plot(self.pulse_cool)
        plot(self.shape_cool*20000)
        legend(["Xpdc", "Ypdc", "shape_pdc", "Xcool", "Ycool", "shape_cool"])
    
    def plot_cov(self):
        import mayavi.mlab as mlab
        mlab.figure()
        mlab.barchart(self.cov)
        mlab.axes()
        
    
def analyze():
    mcs = []
    import mayavi.mlab as mlab
    
    for gamma_b in linspace(2*pi*1e3, 2*pi*17e3, 5):
        mc = MonteCarlo()
        mc.run_stats(35.5e-6, 0.1e-6, 300, gamma_b=gamma_b)
        mcs.append(mc)
        mlab.figure()
        mlab.barchart(mc.cov)
        mlab.axes()
        mlab.title("gamma_b/2pi = "+str(gamma_b/(2*pi)))
    return mcs
        