"""
A package to recover the pose and focal length of a camera given 3D
pionts and their corresponding 2D projections. Includes functions that
map from 3D to 2D projections and their derivatives (that's the core
part).

The camera has seven parameters (wx,wy,wz,tx,ty,tz,f). (wx,wy,wz)
describe its rotation about the origin as parameters of a Cayley
rotation (see the docsring on rot_cayley).  (tx,ty,tz) describe its
translation about the origin. The last parameter is its focal length.

With R(w) denoting the rotation matrix and T=(tx,ty,tz) denoting the
translation vector, the camera transforms a point (x,y,z) into image
points (u,v) as

  (x',y',z') = R(w) (x,y,z) + T

Then

 (u,v) = (f x'/z' + center_u,   f y'/z' + center_v)

The center of projection (center_u,center_v) is assumed known (I use
the center of the image when I use this package).

I used Cayley rotations instead of twists or Euler angles because I've
been curious about them for years.  They turn out to simplify the
code.
"""

from numpy import *

def PnP(x,y,z,u,v,cam,center, w=1., rendering=True):
    """Finds camera parameters (pose and focal length) that minimize
    the projection error between real-world points and their
    cooresponding 2D image points. An initial pose must be supplied.


    x,y,z are vectors of x,y,and z real world coordinates
    respectively.  u,v are vectors of u,v are vectors of image
    coordinates. all these vectors must have the same length. cam is
    an initial guess of the camera parameters. it is a vector of
    length 7.  center is the center of projection.  the reprojection
    error for each point is optionally re-weighted by w.

    rendering is passed directly to fmin_nlls.
    """
    wsqrt = sqrt(w)

    def _func(a):
        A,dA = matrix_rigid3d(a, deriv=True)
        uu,vv,duu,dvv = transform_point(x,y,z,A,dA)
        F = hstack((wsqrt*uu,
                    wsqrt*vv))

        dF = hstack((wsqrt*duu,
                     wsqrt*dvv ))

        return F,dF

    cam = fmin_nlls(_func,
                    hstack((wsqrt*(u-center[0]),
                            wsqrt*(v-center[1]))),
                    cam,
                    rendering=rendering)

    return cam



def transform_point(x,y,z,A,dAdp=None):
    """Apply a 3x4 linear transform A to points in x,y,z and project
    the resulting points back.

    If dA/dp is specified, also returns the derivatives of the points
    wrt dp. In that case dAdp must be a sequence, of partial derivates
    of A with respect to different scalar parameters.  So if A is a
    function of three scalars p1,p2,p3, dAdp is a sequence of 3x4
    matrices dA/dp1, dA/dp2, dA/dp3.

    Specifically, compute

       [u;v;1] = P A [x;y;z;1]

    where P y = y/y_3 is a projection operator.

    For convenience, define xyz=[x;y;z;1] and g = 1/(A_3 [x;y;z;1]).

    The derivative of u and v wrt p is

       du/dp = (dA_1/dp) xyz g - (A_1 xyz) g^2 (dA_3/dp xyz)
             = g  ( (dA_1/dp) - u (dA_3/dp) )  xyz
       dv/dp = g  ( (dA_2/dp) - v (dA_3/dp) )  xyz
    """
    # number of points
    n = 1 if isscalar(x) else len(x)

    xyz = vstack((x, y, z, ones(n)))

    # apply the linear transform
    Xt = dot(A,xyz)

    # apply the projection
    g = 1/Xt[2,:]
    u,v = Xt[:2,:] * g

    if dAdp is None:
        return u,v

    dudp = []
    dvdp = []
    for i,dAdp in enumerate(dAdp):
        # du/dp
        dudp.append( g * dot(dAdp[0,:], xyz) - g * u * dot(dAdp[2,:], xyz) )
        # dv/dp
        dvdp.append( g * dot(dAdp[1,:], xyz) - g * v * dot(dAdp[2,:], xyz) )

    return u,v,dudp,dvdp


# the skew symmetrix bases
S0 = array([[0,0,0],
            [0,0,-1],
            [0,1,0]])
S1 = array([[0,0,1],
            [0,0,0],
            [-1,0,0]])
S2 = array([[0,-1,0],
            [1,0,0],
            [0,0,0]])



def rot_cayley(w,deriv=False):
    """Cayley rotation and its partial derivatives.

    Let S=skew(w) be the skew symmetric matrix associated with w.  A
    Cayley rotation R=(I+S)(I-S)^-1 is a rotation about w. Unlike
    exponential maps, however, the amount of rotation is not given by
    ||w||, but by the angle of the complex number (1-||w||^2) + i 2
    ||w||.

    To see this, write the eigendecomposition of S as S=PDP^-1. We
    know S*w=0 (because S is a cross product matrix for w), so w is an
    eigenvector of S with eigenvalue 0. The other eigenalues are
    ||w||i and -||w||i by the sine formula for cross products.

    R, on the other hand, has eigendecompostion

         R=P [(D+I)(I-D)^-1] P^-1

    So S and R have the same eigenvectors. If l(S) is an eigenvalue of
    S, l(R) is given by

         l(R) = (l(S)+1) / (1-l(S))

    This implies w is an eigenvector of R with eigenvalue 1: R leaves
    w intact and is therefore a rotation about it.

    The other eigenvalues are

         (1 + ||w||i) /  (1-||w||i)
       = ( 1-||w||^2  + 2 ||w|| i ) / (1+||w||^2),

    which, as expected, has modulus 1. Its argument is
        th = atan( 2 ||w|| / ( 1-||w||^2)
    so that the remaining eignevalues are

        l(R) = exp(i th), l(R) = e(-i th)
    """
    x,y,z = w

    D = 1+x**2+y**2+z**2
    R = array([[1+x**2-y**2-z**2,      2*x*y-2*z    ,  2*y+2*x*z],
               [2*x*y+2*z       ,   1-x**2+y**2-z**2,  2*y*z-2*x],
               [2*x*z-2*y       ,      2*x+2*y*z    ,  1-x**2-y**2+z**2]])

    if deriv:
        Rx = array([[2*x,    2*y,  2*z],
                    [2*y,   -2*x, -2],
                    [2*z,    2,   -2*x]])
        dRdx = Rx/D -2*x/D**2 * R
        Ry = array([[-2*y,   2*x,   2  ],
                    [ 2*x,   2*y,   2*z],
                    [-2   ,  2*z,  -2*y]])
        dRdy = Ry/D -2*y/D**2 * R
        Rz = array([[-2*z,   -2,    2*x],
                   [ 2  ,   -2*z,   2*y],
                   [ 2*x,    2*y,   2*z]])
        dRdz = Rz/D -2*z/D**2 * R
        return R/D, dRdx,dRdy,dRdz
    else:
        return R/D


def rot_to_cayley(R):
    """Recover Cayley parameters of a rotaton matrix.

    R = (I+S)*inv(I-S), where S is the skew-symmetric matrix
    associated with w. Find S by writing

       R (I-S)  = I+S
       R-I = S+RS = (I+R) S
       S = inv(I+R)(R-I)
    """
    I = eye(3)
    S = linalg.solve(I+R, R-I)
    # unskew the matrix
    w = S[(2,0,1),(1,2,0)]
    return w


def matrix_rigid3d(a,deriv=False):
    """Return a 3x4 matrix that applies the 3D rigid-body
    transformation and scaling given by a=(wx,wy,wz,tx,ty,tz,f) to a
    point in the XY plane at z=1.  Optionally returns the derivatives
    of the marix wrt a.

    The matrix acts on a vector [x;y;z;1] as follows:

        A [x;y;z;1] = diag([f f 1]) (R [x;y;z;1] + T)
                    = F (R [x;y;z;1] + T)
                    = F (R + [0 T]) [x;y;z;1]

    The derivatives of A wrt to a are
        dA/dtx = [0 0 0 f; 0 0 0 0; 0 0 0 0]
        dA/dty = [0 0 0 0; 0 0 0 f; 0 0 0 0]
        dA/dtz = [0 0 0 0; 0 0 0 0; 0 0 0 1]
        dA/dwi = F [dR/dwi, 0]
        dA/df  = diag([1 1 0]) (R + [0 T])
    """
    # parse parameters
    wx,wy,wz,tx,ty,tz,f = a

    # the rotation matrix
    if deriv:
      R,dRdwx,dRdwy,dRdwz = rot_cayley((wx,wy,wz), deriv)
    else:
      R = rot_cayley((wx,wy,wz), deriv)

    # A = F (R + [0 T])
    A = zeros((3,4))
    A[:,:3] = R
    A[:,-1] = tx,ty,tz
    A[:2,:] *= f

    if deriv:
      # derivatives wrt translation
      dAdtx = zeros((3,4))
      dAdtx[0,-1] = f
      dAdty = zeros((3,4))
      dAdty[1,-1] = f
      dAdtz = zeros((3,4))
      dAdtz[2,-1] = 1
      # derivatives wrt focal
      dAdf = zeros((3,4))
      dAdf[:,:3] = R
      dAdf[:,-1] = tx,ty,tz
      dAdf[-1,:] = 0
      # derivatives wrt rotation
      dAdwx = zeros((3,4))
      dAdwx[:,:3] = dRdwx
      dAdwx[:2,:] *= f
      dAdwy = zeros((3,4))
      dAdwy[:,:3] = dRdwy
      dAdwy[:2,:] *= f
      dAdwz = zeros((3,4))
      dAdwz[:,:3] = dRdwz
      dAdwz[:2,:] *= f

      return A, (dAdwx,dAdwy,dAdwz,dAdtx,dAdty,dAdtz,dAdf)
    else:
      return A


def test_matrix():
  cam = random.randn(7)

  dcam = random.randn(len(cam))
  dcam /= nor(dcam) * 1e8

  A,dA = matrix_rigid3d(cam, deriv=True)

  A1 = matrix_rigid3d(cam+dcam)

  A1_est = A + sum(dcam.reshape((-1,1,1)) * dA, axis=0)
  print ' (A1_est-A1)/||A-A1||:', nor(A1_est-A1)/nor(A-A1)



def deriv_check(func,x,h=1e-3,i=None,verbose=False):
    """Display the defect between the numeric derivative of func and
    its analytic derivatve along a random direction"""
    if i is None:
        dx = random.randn(len(x))
        dx *= h / nor(dx)
    else:
        dx = zeros(len(x))
        dx[i] = h

    f0,df0 = func(x)
    f1,_ = func(x + dx)

    # the directional derivative along dx
    if verbose:
        print 'analytic df:', nor(dot(dx,df0))
        print 'numeric df:', nor(f1-f0)
        print '|numeric - analytic|:', nor(dot(dx,df0)- (f1-f0))
    print '|df_anal - df_num| / |df_num|:', nor(dot(dx,df0)- (f1-f0)) / nor(f1-f0)


def fmin_nlls(func,y,a,maxlinesearch=10,rendering=True):
    """Nonlinear Least Squares using a Levenberg approximation.

    Solves

          minimize_a   sum_i ( f_i(a) - y_i )^2

    The function f returns a vector of the same length as the vector
    y.

    Does this by approximaging each term by linearizing f over a:

         f_i(a) = f_i(a0) + d/da f_i(a) (a-a0)

    and solving for (a-a0) with least squares.


    If rendering is a function, it is called at each iteration, to
    provide the caller a progress report. Otherwise, if it evaluates
    to true, prints a progress report for each iteration. If it
    evaluates to false, does nothing.
    """
    # the current function value and its derivative
    f,df = func(a)
    # current loss
    loss = sum( (f-y)**2 )

    nsteps = 0
    step = 1.

    linesearch_ok = True
    while linesearch_ok:
        nsteps += 1

        # notify the caller or print
        if hasattr(rendering, '__call__'):
            # rendering may optionally perturb a
            rendering(nsteps, a, loss, f)
        elif rendering:
            print 'nsteps:' , nsteps, 'loss:', loss

        # search direction
        da = linalg.lstsq(df.T, y-f)[0]
        # print 'da:', da

        # aggressively increase the step size
        step *= 1.1

        # linesearch along the latest gradient
        linesearch_ok = False
        for it in xrange(maxlinesearch):
            # the proposed step
            anew = a + step*da
            # the value of the step
            fnew,dfnew = func(anew)

            newloss = sum( (fnew-y)**2 )

            if rendering!=False:
                print '   ', it, 'step:',step, 'loss:', loss, 'newloss:', newloss

            if newloss < loss and abs((loss-newloss)/loss) > 1e-6:
                # accept the step if the new loss is significantly smaller.
                a = anew
                f = fnew
                df = dfnew
                loss = newloss
                linesearch_ok = True
                break
            else:
                # bad step. take a smaller step and try again
                step *= 0.5

    return a
