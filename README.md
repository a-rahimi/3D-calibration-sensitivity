Determining the Position of a Camera with a (Planar) Calibtration Target
=======

A straightforward calibration procedure that accepts 3D points and their 2D projections and estimates a camera pose and focal length that would minimize the reprojection error of the 3D points. This is similar to OpenCV's solvePnP. It works even when the calibration target is planar.

The included [IPython notebook](http://nbviewer.ipython.org/github/a-rahimi/3D-calibration-sensitivity/blob/master/Sensitivity.ipynb) examines its sensitivity under image noise and perturbations of the initial guess of the pose.
