import numpy as np
from multiprocessing import Pool
from timeit import default_timer as timer
from ddgclib._particle_liquid_bridge_flo import *


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from timeit import default_timer as timer
from multiprocessing import Manager, freeze_support

# Initialisieren Sie den Manager, um gemeinsam genutzte Listen zu erstellen
manager = Manager()
v_rel_list = manager.list()
time_list = manager.list()


def fun_V_analytic(diameter, length):
    V_an = np.pi / 4 * diameter ** 2 * length
    return V_an


def fun_iterable_2(args):
    refinement, tau, t_f, diameter, length = args

    global v_rel_list
    global time_list

    gamma = 0.0728
    print(f'refinement = {refinement}')

    starttime = timer()
    v_l = 0 - length / 2  # lower length-coordinate
    v_u = 0 + length / 2  # higher length-coordinate
    V_an = fun_V_analytic(diameter, length)

    dummy_parameter = fun_liquid_bridge(refinement, v_l, v_u, tau, t_f, diameter, gamma)
    HC = dummy_parameter[0]

    V_num = 0
    for v in HC.V:
        V_ijk = volume(v)
        V_num += np.sum(V_ijk)

    V_num = V_num / 12
    v_rel = ((V_an - V_num) / V_an) * 100

    endtime = timer()

    print(f"Time elapsed: {endtime - starttime:.2f} s für refinement = {refinement}")
    print(f"Volume calculated: {V_num:.2f} m^3 für refinement = {refinement}")
    print(f"Volume deviation: {v_rel:.2f} % für refinement = {refinement}")

    # Hier verwenden Sie Locks, um sicherzustellen, dass der Zugriff auf die Listen koordiniert wird
    with manager.Lock():
        v_rel_list.append(v_rel)
        time_list.append(endtime - starttime)

        if 1:
            string_savename = 'd' + str(diameter) + 'l' + str(length) + 'ref' + str(refinement)
            np.savetxt(string_savename + '_vrel' + '.txt', v_rel_list)
            np.savetxt(string_savename + '_time' + '.txt', time_list)


if __name__ == '__main__':
    # Fügen Sie freeze_support() hinzu, um Probleme bei Windows zu vermeiden
    freeze_support()

    refinement_end = 5
    tau = 0.1
    t_f = 0
    diameter = 3
    length = 1

    tau_value = tau
    t_f_value = t_f
    length_value = length
    diameter_value = diameter

    refinement_values = np.arange(2, refinement_end, 1)
    print(refinement_values)

    max_workers = 2

    args_list = [(refinement, tau_value, t_f_value, length_value, diameter_value) for refinement in refinement_values]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Verwenden Sie as_completed, um auf die abgeschlossenen Ergebnisse zuzugreifen
        futures = [executor.submit(fun_iterable_2, args) for args in args_list]
        for future in as_completed(futures):
            # Warten Sie auf den Abschluss des Prozesses
            result = future.result()
            print(result)

    print('Ende')