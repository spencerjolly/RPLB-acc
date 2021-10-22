# RPLB-acc
Scripts and functions for accelerating single electrons with ultrafast radially-polarized laser beams (RPLBs) including with spatial-temporal couplings (STCs). So far there are three cases considered: without any STCs, with longitudinal chromatism, and with spatial chirp.

The function 'RPLB_NoSTC.py' produces the final kinetic energy of an electron depending on the laser parameters (without any STCs) and the initial conditions of the electron. The script (jupyter notebook) 'NoSTC_OnAxis.ipynb' shows how to call this function in a nested loop, optimizing over the CEP of the laser and the initial electron position, reproducing results from Ref. [1]. The function 'RPLB_acc_NoSTC_2D.py' produces the time, z position, r position, and final kinetic energy of an electron accelerated including off-axis and non-paraxial fields of the laser and off-axis positions of the electron.

Functions 'RPLB_LC.py' and 'RPLB_acc_LC_2D.py' produce the same results as above for the case where the laser has longitudinal chromatism (LC) in the focus. The script (jupyter notebook) 'LC_OnAxis.ipynb' shows how to call this function in a nested loop, optimizing over the CEP of the laser, the initial electron position, and the magnitude of the LC (tau_p parameter), reproducing results from Ref. [2].

The function 'RPLB_acc_SC_2D.py' produces the same results as above for the case where the laser has spatial chirp (SC) in the focus. Because of the breaking of cylindrical symmetry there is no 1D scenario in this case. An electron starting initially off-axis will still gain transverse momentum and move off-axis.

The script (jupyter notebook) 'Function_test.ipynb' shows simple examples of how to call all of the functions.

The general model used to produce the fields simulated here can be found in Ref. [4] and in the repository RPLB-STC (https://github.com/spencerjolly/RPLB-STC).

<b>Under development:</b>

-functions do not currently automatically choose the starting time, ending time, or the temporal resolution

-there is not yet a script reproducing the results from Ref. [3].

<b>References:</b>

[1] L. J. Wong and Franz X. Kärtner, Direct acceleration of an electron in infinite vacuum by a pulsed radially-polarized laser beam, Optics Express <b>18</b>, 25035-25051 (2010). https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-24-25035&id=208171

[2] S. W. Jolly, Influence of longitudinal chromatism on vacuum acceleration by intense radially polarized laser beams, Optics Letters <b>44</b>, 1833–1836 (2019). https://www.osapublishing.org/ol/abstract.cfm?uri=ol-44-7-1833

[3] S. W. Jolly, On the importance of frequency-dependent beam parameters for vacuum acceleration with few-cycle radially-polarized laser beams, Optics Letters <b>45</b>, 3865–3868 (2020). https://www.osapublishing.org/ol/abstract.cfm?uri=ol-45-14-3865 

[4] S. W. Jolly, Focused fields of ultrashort radially polarized laser pulses having low-order spatiotemporal couplings, Physical Review A 103, 033512 (2021). https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.033512 (https://arxiv.org/abs/2012.02729)
