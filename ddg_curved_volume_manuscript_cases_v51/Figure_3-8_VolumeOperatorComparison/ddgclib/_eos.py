#from CoolProp.CoolProp import PropsSI

# Equation of state for water droplet:
def eos(P=101.325, T=298.15):
    # P in kPa T in K
    return PropsSI('D','T|liquid',298.15,'P',101.325,'Water') # density kg /m3

# Surface tension of water gamma(T):
def IAPWS(T=298.15):
    T_C = 647.096  # K, critical temperature