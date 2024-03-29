# RPLB-acc
Scripts and functions for accelerating single electrons with ultrafast radially-polarized laser beams (RPLBs) including with spatial-temporal couplings (STCs). So far there are four cases considered: without any STCs, with longitudinal chromatism, with a frequency-varying beam size (g0), and with spatial chirp.

The function 'RPLB_acc_NoSTC.py' produces the kinetic energy of an electron depending on the laser parameters (without any STCs) and the initial conditions of the electron. The script (jupyter notebook) 'NoSTC_OnAxis.ipynb' shows how to call this function in a nested loop, optimizing over the CEP of the laser and the initial electron position, reproducing results from Ref. [1]. The function 'RPLB_acc_NoSTC_2D.py' produces the time, z position, r position, and kinetic energy of an electron accelerated including off-axis and non-paraxial fields of the laser and off-axis positions of the electron.

Functions 'RPLB_acc_LC.py' and 'RPLB_acc_LC_2D.py' produce the same results as above for the case where the laser has longitudinal chromatism (LC) in the focus. The script (jupyter notebook) 'LC_OnAxis.ipynb' shows how to call this function in a nested loop, optimizing over the CEP of the laser, the initial electron position, and the magnitude of the LC (tau_p parameter), reproducing results from Ref. [2].

Functions 'RPLB_acc_g0.py' and 'RPLB_acc_g0_2D.py' produce the same results as above for the case where the laser has frequency-varying beam parameters according to the 'Porras factor' g_0. These functions can reproduce results from Ref. [3].

The function 'RPLB_acc_SC_2D.py' produces the same results as above for the case where the laser has spatial chirp (SC) in the focus. Because of the breaking of cylindrical symmetry there is no possible 1D scenario in this case. An electron starting initially on-axis will still gain transverse momentum and move off-axis when there is non-zero SC. The 3D script 'RPLB_acc_SC_3D.py' simulates an arbitrary electron trajectory in full dimensions when there is SC (i.e. the y-component of the force is added). These functions have not yet been used to produce scientific results.

Finally, there are multiple functions that apply all of the STCs that are possible (in this formalism) for a given dimensionality. The function 'RPLB_acc_anySTC.py' simulates electron acceleration purely on-axis (r=0) for a pulse that has LC and/or g_0 considered (i.e. retaining cylindrical symmetry). The function 'RPLB_acc_anySTC_2D.py' simulates the 2D case, where SC is added to LC and/or g_0 (simulating electrons in the x-z plane only). The 3D script 'RPLB_acc_anySTC_3D.py' simulates an arbitrary electron trajectory in full dimensions with LC, g_0, and/or SC.

The script (jupyter notebook) 'Function_test.ipynb' shows simple examples of how to call many of the functions. There are numerous other scripts that test and compare specific functions.

The general model used to produce the fields simulated here is from Ref. [4]. The model that expands to include STCs can be found in Ref. [5] and in the repository RPLB-STC (https://github.com/spencerjolly/RPLB-STC).

<b>Under development:</b>

-There are functions with 'April' in their name that use the model from [6] to allow for arbitrarily non-paraxial focusing. These are essentially copies of the other functions using the non-paraxial model from [4] that is not accurate to arbitrarily tight focusing. There are numerous scripts that test these functions and compare them to the model in the other "standard" functions. These 'April' functions work well, but are not as fully tested as others.

-Not all functions currently automatically choose the starting time, ending time, or the temporal resolution of the finite difference method. This means that they may not work as-is for very high laser powers or very tight focusing (i.e. there is a limit on the laser intensity and the associated high electron energy).

-The functions are all limited at the moment to Gaussian spectral profiles (except for the April model cases), effectively limiting the minimum pulse duration that can be properly simulated. For example with a wavelength of 800 nm the lower limit of the pulse duration in the g_0 functions is around 3-4 fs. There are other approximations in the LC and SC functions such that the pulse must be longer than few-cycle, limiting the applicablity of those functions to pulses as short as ~10 fs for an 800 nm wavelength.

-There are functions entitled 'RPeLG_acc_NoSTC.py' and 'RPLG_acc_NoSTC.py', and functions that use the 'April' model with STCs added. All of these functions are very experimental and should not necessarily be considered correct.

<b>References:</b>

[1] L. J. Wong and Franz X. Kärtner, Direct acceleration of an electron in infinite vacuum by a pulsed radially-polarized laser beam, Optics Express <b>18</b>, 25035-25051 (2010). https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-18-24-25035&id=208171 (open access)

[2] S. W. Jolly, Influence of longitudinal chromatism on vacuum acceleration by intense radially polarized laser beams, Optics Letters <b>44</b>, 1833–1836 (2019). https://www.osapublishing.org/ol/abstract.cfm?uri=ol-44-7-1833 (https://arxiv.org/abs/1811.09412)

[3] S. W. Jolly, On the importance of frequency-dependent beam parameters for vacuum acceleration with few-cycle radially-polarized laser beams, Optics Letters <b>45</b>, 3865–3868 (2020). https://www.osapublishing.org/ol/abstract.cfm?uri=ol-45-14-3865 (https://arxiv.org/abs/1912.04026)

[4] Y. I. Salamin, Fields of a radially polarized Gaussian laser beam beyond the paraxial approximation, Optics Letters <b>17</b>, 2619–2621 (2006). https://opg.optica.org/ol/abstract.cfm?uri=ol-31-17-2619

[5] S. W. Jolly, Focused fields of ultrashort radially polarized laser pulses having low-order spatiotemporal couplings, Physical Review A 103, 033512 (2021). https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.033512 (https://arxiv.org/abs/2012.02729)

[6] A. April, Ultrashort, Strongly Focused Laser Pulses in Free Space, chapter 16 in the book "Coherence and Ultrashort Pulse Laser Emission", 355-382 (2010). https://www.intechopen.com/chapters/12677

<b>Acknowlegements:</b>

The initial ideas and publications associated with this work were done while S.W.J. was at CEA-Saclay from January 2018 until November 2019. From October 2021 this work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 801505 (IF@ULB Cofund PostDoc grant for S.W.J.).
