#Ianto Cannon 2025 Jun 2. Plot the detachment size of pinned and spreading bubbles
"""
Bo>2pi*R**3/V
V>2pi*r*R**2/Bo
Chesters: V>2pi*r*sigma/rho/g where r is the contact radius
4pi*R**3/3>2pi*r*sigma/rho/g
Demirkir: 2R_bubble**3/3>r*sigma/rho/g
R_bubble**3>3r*lambda**2/2
R_bubble/lambda>(3r/lambda/2)**(1/3)

Bo=rho*g*R**2/sigma where R is the top radius

Plot contact angle on x and Bo_c on y
Plot r on x axis and Bo_c on y
Plot V/lambda**3 on y and r/lambda on x
Plot R_bubble/lambda on y and r/lambda on x
Set r, V, rho. Does it detach?
People want to know what is the bubble volume that will come from a cavity
Bo_cavity=rho*g*r**2/sigma=r**2/lambda**2
Bo_bubble=rho*g*R_bubble**2/sigma=R**2/lambda**2
Where V=4pi*R_bubble**3/3

4pi*R_bubble**3/3>2pi*r*sigma/rho/g
4pi*R_bubble**3/3>2pi*r*lambda**2 where capillary length lambda**2=sigma/rho/g
4pi*R_bubble**3/lambda**3/3>2pi*r/lambda
R_bubble**3/lambda**3>3*r/lambda/2

Can I get a good fit without the foot pressure?

Choose R_top, and lambda, use Adams Bashforth integration to get V, R_bubble, and r with contact and angle 90. 
"""
import numpy as np
from ddgclib._plotting import plot_polyscope, plot_detach_profile, plot_detach_radius_vs_cont_angle, plot_detach_radius_vs_cont_radius
from ddgclib._bubble import AdamsBashforthProfile, load_complex

#Parameters
RadTop = 1 # m, radius of curvature of bubble top
prm = {} # dictionary of parameters
prm['contactAng'] = -1 #radians, angle inside the spherical cap. Set negative for pinned contact line
prm['gamma'] = 1 # N/m, surface tension
prm['gravity'] = 1 # m/s^2 gravitational acceleration
anglePrev=0
RadMax=0
BoMax=1
if True:
  fname='data/fritz0.txt'
  with open(fname, "w") as fritz_txt:
    print('saving',fname)
    for Bo in [.56,0.6064,1.27,2**14]:
      #AdamsBashforthProfile(BoPrev, RadTop, .5*np.pi, fname=f'data/pin{b-1}.txt')
      #AdamsBashforthProfile(BoPrev, RadTop, fname=f'data/spread{b-1}.txt')
      VPin, RadFootPin, heightPin, centroidPin, anglePin = AdamsBashforthProfile(BoT, RadTop, .5*np.pi, fname=f'data/pin{BoT}.txt')
      RadPin = (3*VPin/4/np.pi)**(1/3)
      VSpr, RadFootSpr, heightSpr, centroidSpr, angleSpr = AdamsBashforthProfile(BoT, RadTop, fname=f'data/spread{BoT}.txt')
      RadSpread = (3*VSpr/4/np.pi)**(1/3)
      print(BoT, angleSpr, RadSpread/capiLen, RadFootSpr/capiLen, RadFootPin/capiLen, RadPin/capiLen, file=fritz_txt)
      #AdamsBashforthProfile(Bo, RadTop, .5*np.pi, fname=f'data/pin{b}.txt')
      #AdamsBashforthProfile(Bo, RadTop, fname=f'data/spread{b}.txt')
if False:
  fname='data/fritz.txt'
  with open(fname, "w") as fritz_txt:
    print('saving',fname)
    for b, Bo in enumerate(np.logspace(-10, 10, num=1001, base=2)):
    #for b, Bo in enumerate(np.logspace(np.log2(1.25), np.log2(1.32), num=1001, base=2)):
      spreadName=None
      pinName=None
      if not np.log2(Bo)%1: 
        print('b',b,'Bo',Bo,'1/Bo',1/Bo)
        spreadName=f'data/spread{b}.txt'
        pinName=f'data/pin{b}.txt'
      prm['density'] = Bo*prm['gamma']/prm['gravity']/RadTop**2 # kg/m3, bubble density difference
      capiLen = ( prm['gamma'] / prm['density'] / prm['gravity'] )**.5
      VPin, RadFootPin, heightPin, centroidPin, anglePin = AdamsBashforthProfile(Bo, RadTop, .5*np.pi, fname=pinName)
      RadPin = (3*VPin/4/np.pi)**(1/3)
      VSpr, RadFootSpr, heightSpr, centroidSpr, angleSpr = AdamsBashforthProfile(Bo, RadTop, fname=spreadName)
      RadSpread = (3*VSpr/4/np.pi)**(1/3)
      if False and (anglePrev-np.pi/2)*(angleSpr-np.pi/2) < 0:
        AdamsBashforthProfile(Bo, RadTop, .5*np.pi, fname=f'data/pin{b}.txt')
        AdamsBashforthProfile(Bo, RadTop, fname=f'data/spread{b}.txt')
        AdamsBashforthProfile(BoPrev, RadTop, .5*np.pi, fname=f'data/pin{b-1}.txt')
        AdamsBashforthProfile(BoPrev, RadTop, fname=f'data/spread{b-1}.txt')
      anglePrev = angleSpr 
      BoPrev = Bo
      print(Bo, angleSpr, RadSpread/capiLen, RadFootSpr/capiLen, RadFootPin/capiLen, RadPin/capiLen, file=fritz_txt)
      if RadMax < RadSpread/capiLen:
        RadMax = RadSpread/capiLen
        BoMax = Bo
    #AdamsBashforthProfile(BoMax, RadTop, fname=f'data/pin_spreadMax.txt')
plot_detach_radius_vs_cont_angle()
plot_detach_radius_vs_cont_radius()
plot_detach_profile()
