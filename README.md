Determining the Position of a Camera with a (Planar) Calibtration Target
=======

This package estimates a camera's pose and its focal length from a set
of 3D points and their corresponding 2D image locations.  The
underlying algorithm finds the camera parameters that minimize the
reprojection error of the 3D points. This is similar to OpenCV's
solvePnP routine. It works even when the calibration target is planar
and is written in Python.

The included
[IPython notebook](http://nbviewer.ipython.org/github/a-rahimi/3D-calibration-sensitivity/blob/master/Sensitivity.ipynb)
examines the sensitivity of this routine for different planar
calibration targets.
