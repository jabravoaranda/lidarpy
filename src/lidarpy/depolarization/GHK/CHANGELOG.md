# Changelog

## [Unreleased]
- Moved changes to a separate CHANGELOG file.
- Added EUPL licence file.
- Renamed script files to lidar_correction_ghk*.

## 9-PollXT 
- Same as before.

## 9-Ralph-6 
- For POLLY_XT   RP = 1 - TP  RS = 1 - TS

## 9
- Output Infos to Console and File

## 8c-Ralph-6 
- For POLLY_XT   RP = 1 - TP  RS = 1 - TS

## 8c 
- Tp, Ts, Rp, Rs individually with individual errors

## 8b 
- more code comments

## 8a 
- output Itotal (F11) with error

## 7b 
- cosmetic changes: YesNo function, plot title, warning if N is too large, elapsed time; equation source: rotated_diattenuator_X22x5deg.odt

## 7a 
- Bugfix:  when NOT explicitly varying LDRCal, not LDRCal0 is used, but the last LDRCal = 0.45 from the previous loop over LDRCal to determine K(LDRCal)

## 7
- just a new main version for ver6i - now saving LDRMIN - LDRMAX to file

## 6i 
- Several bugs fixed: => most GH equations newly formulated.  use ver6e inputs

## 6h 
- Trying to correct the absolute values of GH

## 6g 
- varying LDRCal and K calculated for assumed setup (input ver6e)

## 6f 
- angles from degree to rad before loop ( only 2% less processor time)

## 6e 
- varying LDRCal

## 6d 
- plots also with iPython and python command prompt (under Anaconda at least)

## 6c 
- rotated Pol-Filters and TypeC = 6 for all LocC; QWP calibrator; resorting of code; correct rotator calib without rot-error;

## 6b 
- ?

## 6
- with rotated Pol-Filters behind the PBS + some vector equations, only for Loc = 3
- todo: correct unpol transmittance; compare with ver 5a.

## 5a 
- with Type = 6 : retarding diattenuator at +-22.5° (with 180° retardance = HWP), first vector equations

## 4c5c 
- ?

## 4c6.py
- colored hist overlays for certain parameters in function

## 4c5.py 
- colored hist overlays for certain parameters

## 4c4.py 
- incl. PollyXT with ER error

## 4c3.py 
- incl. loop over LDRtrue with plot of errors

## 4c2.py 
- S2g Bug in B fixed
- is faster (9 instead of 16 sec) but less clear code

## 4c.py 
- function and for loop split to speed up the code 09.07.16, vf

## 4b.py 
- with function 09.07.16, vf

## 4a.py 
- error loops 09.07.16, vf

## 3c.py 
- some code lines moved in the if structures and combined at end => now the option
  to remove the rotational error epsilon for normal measurements works 09.07.16, vf

## 3b 
- several bugs fixed 08.07.16, vf

## 3a 
- option to remove the rotational error epsilon for normal measurements  08.07.16, vf

## 2c 
- with output of input values

## 2b 
- with output to text.file