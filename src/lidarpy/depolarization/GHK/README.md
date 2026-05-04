Calculation of polarisation correction factors for atmospheric lidar system, developed by Volker Freudenthaler (LMU, Munich, Germany).

# Theory
The theoretical basis of the script is described in detail in :

Freudenthaler, V.: 
About the effects of polarising optics on lidar signals and the Δ90 calibration,
Atmos. Meas. Tech., 9, 4181-4255, doi:10.5194/amt-9-4181-2016, 2016
http://www.atmos-meas-tech.net/9/4181/2016/

Additional information can be found in:

Bravo-Aranda, J. A., Belegante, L., Freudenthaler, V., Alados-Arboledas, L., Nicolae, D., Granados-Muñoz, M. J.,
Guerrero-Rascado, J. L., Amodeo, A., D'Amico, G., Engelmann, R., Pappalardo, G., Kokkalis, P., Mamouri, R.,
Papayannis, A., Navas-Guzmán, F., Olmo, F. J., Wandinger, U., Amato, F., and
Haeffelin, M.: 
Assessment of lidar depolarization uncertainty by means of a polarimetric lidar simulator, 
Atmos. Meas. Tech., 9, 4935-4953, doi:10.5194/amt-9-4935-2016, 2016.
http://www.atmos-meas-tech.net/9/4935/2016/

# Use
To run the script you need to:

1. Read the information in the script header. 

1. Create a file describing your system settings and parameters. You can find an example file in the "system_settings"
   folder. Give a descriptive name and save it in the "system_settings" folder.

2. Edit the "GHK_0.9.8e5_Py3.7.py" or the current script file with an ASCII editor and set the variable InputFile below line 281 to the filename you chose in step 1.

3. Run the script "GHK_0.9.8e5_Py3.7.py" (or the current version).

Note: _0.9.8e5 in the script filename is the script version and _Py3.7 indicates that it is tested with Python 3.7. 
 
