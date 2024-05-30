import numpy as np
from multiprocessing import Pool
from timeit import default_timer as timer
from ddgclib._particle_liquid_bridge_flo import *
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager


v_rel_list  = []
time_list   = []


def fun_V_analytic(diameter, length):
    V_an = np.pi/4*diameter**2*length
    return V_an

def save_complex(HC,filename):
    for v in HC.V:
       # print('---')
        v_new = []
        for i, vi in enumerate(v.x):
            #print(type(vi))
            v_new.append(float(vi))
            #v.x[i] = float(vi)
        HC.V.move(v, tuple(v_new))
        v.x_a = np.array(v.x_a, dtype='float')
        #v.x_a = v.x_a.astype(float)
        #for vi in v.x:
         #   print(type(vi))
    #print(type(v.x_a))
    HC.save_complex(fn = filename)

def fun_iterable_2( refinement, tau= 0.1, t_f=0, diameter=1, length = 1):
    gamma=0.0728
    #v_rel_list = []
    #time_list = []
    print(f'refinement = {refinement}')

    starttime = timer()
    v_l         = 0 - length/2  # lower length-coordinate
    v_u         = 0 + length/2   # higher length-coordinate
    V_an = fun_V_analytic(diameter, length)

   # dummy_parameter = fun_liquid_bridge(refinement,v_l=v_l, v_u=v_u,tau=tau, t_f = t_f,diameter = diameter, gamma = gamma)
    dummy_parameter = fun_liquid_bridge(v_l, v_u,tau, t_f,diameter, refinement,gamma)
    HC = dummy_parameter[0]
    '''
    Volume calculation
    '''
    V_num = 0
    for v in HC.V:
        V_ijk = volume(v)
        #print(V_ijk)
        V_num += np.sum(V_ijk)

    V_num = V_num/12

    v_rel = ((V_an - V_num)/V_an)*100

    endtime = timer()

    print(f"Time elapsed: {endtime-starttime:.2f} s für refinement = {refinement}")
    print(f"Volume calculated: {V_num:.2f} m^3 für refinement = {refinement}")
    print(f"Volume deviation: {v_rel:.2f} % für refinement = {refinement}")

    v_rel_list.append(v_rel)
    time_list.append(endtime-starttime)
    print(v_rel_list)


    string_savename     = 'd'+ str(diameter) + 'l' + str(length) + 'ref' + str(refinement)
    #save_name = 'Complex_tf5000_tau01_refinment3.json'
    save_name = 'Complex'+ string_savename + '.json'
    np.savetxt(string_savename + '_vrel'+ '.txt',v_rel_list)
    np.savetxt(string_savename + '_time'+ '.txt',time_list)
    save_complex(HC, save_name)

    #np.savetxt(string_savename + 'refinement_list'+ '.txt', refinement_list)

#%%

refinement_end = 8
tau = 0.1
t_f = 0
diameter = 2
length = 1

print('Start')

if __name__ == '__main__':
    tau_value = tau
    t_f_value = t_f
    length_value = length
    diameter_value = diameter

    refinement_values = np.arange(2,refinement_end,1)
    print(refinement_values)

    max_workers = 2

    #args_list = [(refinement, tau_value, t_f_value, length_value, diameter_value) for refinement in refinement_values]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        results = list(executor.map(fun_iterable_2,refinement_values,[tau_value]*len(refinement_values),[t_f_value]*len(refinement_values),[diameter_value]*len(refinement_values),[length_value]*len(refinement_values)))
    # Gib die Ergebnisse aus
    print('Ende')




'''
# Die Funktion, die parallel berechnet werden soll
def f(args):
    x, y, z = args
    return x * y + z

if __name__ == '__main__':
    # Definiere die konstanten Werte für y und z
    y_value = 3
    z_value = 2

    # Definiere die Werte für x als NumPy-Array
    x_values = np.array([1, 2, 3, 4])

    # Definiere die maximale Anzahl der gleichzeitig arbeitenden Prozesse
    max_workers = 3

    # Erstelle einen Pool mit mehreren Prozessen (hier verwenden wir die Anzahl der verfügbaren CPU-Kerne)
    with Pool(max_workers) as pool:
        # Verwende die map-Funktion, um die Funktion parallel für jeden Wert von x auszuführen
        results = pool.map(f, zip(x_values, [y_value]*len(x_values), [z_value]*len(x_values)))

    # Gib die Ergebnisse aus
    print(results)
    

'''