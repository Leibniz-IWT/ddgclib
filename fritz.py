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
from ddgclib._plotting import plot_polyscope, plot_profile, plot_detach_radius_vs_cont_angle, plot_detach_radius_vs_cont_radius
from ddgclib._bubble import AdamsBashforthProfile, load_complex

def bubbleProfile(capLen, RadTop, contactAng=-1, fname=None):
#compute analytical interface shape according to eq 1 of Demirkir2024Langmuir
#Input the Bond number Bo, and the radius of curvature at bubble top; RadTop
#Return the volume of the bubble, radius of the contact patch, height of bubble 
#and height of the centre of mass.
  ds=.001
  psi=0
  r=0
  z=0
  Volume=0
  centroid=0
  #fname='data/adams'+str(Bo)+'.txt'
  if fname: 
    adams_txt = open(fname, "w") 
    print('saving',fname)
  for i in range(int(10/ds)):
    r += ds * np.cos(psi)
    dz = ds * np.sin(psi)
    z += dz
    Volume += np.pi*r**2*dz
    centroid += z*np.pi*r**2*dz
    if fname: print(r*capLen, -z*capLen, psi, file=adams_txt)
    #psi += ds * (2 - Bo*z - np.sin(psi)/r)
    psi += ds * (2*capLen/RadTop - z - np.sin(psi)/r)
    #if 2 - Bo*z - np.sin(psi)/r < 0: break
    #if z < -.4: break
    #if contactAng>0 and psi > contactAng: break
    if contactAng<0 and 2-Bo*z-np.sin(psi)/r < 0: break
    if contactAng>0 and psi < contactAng and 2-Bo*z-np.sin(psi)/r < 0: break
  print('i',i)
  if fname: close(adams_txt)
  centroid /= Volume
  return Volume*capLen**3, r*capLen, z*capLen, (z-centroid)*capLen, psi

#Parameters
RadTop = 1 # m, radius of curvature of bubble top
prm = {} # dictionary of parameters
prm['contactAng'] = -1 #radians, angle inside the spherical cap. Set negative for pinned contact line
prm['gamma'] = 1 # N/m, surface tension
prm['gravity'] = 1 # m/s^2 gravitational acceleration
fname='data/fritz.txt'
if False:
 with open(fname, "w") as fritz_txt:
  print('saving',fname)
  for b, Bo in enumerate(np.logspace(-4,3,num=1000)):  #range(1,1000):
    #Bo=b*.01*10**(b/1.)
    #Bo=.001*b
    if b%100==0: 
      print('b',b,'Bo',Bo)
      spreadName=f'data/spread{b:03}.txt'
      pinName=f'data/pin{b:03}.txt'
    else:
      spreadName=None
      pinName=None
    prm['density'] = Bo*prm['gamma']/prm['gravity']/RadTop**2 # kg/m3, bubble density difference
    #print('density',prm['density'])
    capiLen = ( prm['gamma'] / prm['density'] / prm['gravity'] )**.5
    VPin, RadFootPin, heightPin, centroidPin, anglePin = AdamsBashforthProfile(Bo, RadTop, .5*np.pi, fname=pinName)
    RadPin = (3*VPin/4/np.pi)**(1/3)
    VSpr, RadFootSpr, heightSpr, centroidSpr, angleSpr = AdamsBashforthProfile(Bo, RadTop, fname=spreadName)
    RadSpread = (3*VSpr/4/np.pi)**(1/3)
    print(Bo, angleSpr, RadSpread/capiLen, RadFootSpr/capiLen, RadFootPin/capiLen, RadPin/capiLen, file=fritz_txt)
plot_detach_radius_vs_cont_angle()
plot_detach_radius_vs_cont_radius()
