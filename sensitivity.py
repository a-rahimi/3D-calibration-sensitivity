import pdb
import pylab as P
from numpy import *
import pandas as pd

nor = linalg.norm

class Shape:
  """A shape is a set of 3D points. The segments() routine renders
  these points as a line drawing. Currently, i'm only inspecting 2D
  shapes so the endpoints are only 2D"""

  def endpoints(self):
    raise NotImplementedError

  def segments(self, xy):
    raise NotImplementedError

  def draw(self, ax, xy=None):
    if xy is None:
      xy = self.endpoints()
    xy = array(xy)

    h = P.Line2D( xy[:,0], xy[:,1], marker='o', lw=0, color='k')
    ax.add_artist(h)

    for x0,y0,x1,y1 in self.segments(xy):
      h = P.Line2D([x0,x1], [y0,y1], color='r')
      ax.add_artist(h)


class ShapeTriangle(Shape):
  def endpoints(self):
    return (-.8,0), (.8,0), (0,3.)

  def segments(self, xy):
    ll,lr,la = xy
    return [hstack((ll,la)), hstack((la,lr))]

class ShapeGrid(Shape):
  def endpoints(self):
    X,Y = meshgrid(linspace(-2.,2., 3), linspace(0.,6.,7))
    return zip(X.ravel(), Y.ravel())

  def segments(self, xy):
    X,Y = array(xy).T
    X = X.reshape(7,3)
    Y = Y.reshape(7,3)
    hlines = zip(X[:,0], Y[:,0], X[:,-1], Y[:,0])
    vlines = zip(X[0,:], Y[0,:], X[-1,:], Y[-1,:])
    return hlines+vlines



def test_shape_grid():
  s = ShapeGrid()
  P.clf()
  s.draw(P.gca())
  P.axis([-3,3,-1,5])
  P.draw()

def test_shape_triangle():
  s = ShapeTriangle()
  P.clf()
  s.draw(P.gca())
  P.axis([-3,3,-1,5])
  P.draw()


def test_view(Shape, cam=(1.,0.,0.,  0.,-1.4,4., 300.) ):
  """Draw the shape in robot coordinates, then in image coordiantes
  after it's been projected by the given camera parameters"""

  s = Shape()
  xy = array(s.endpoints())

  # two axes, side by side. one for the shape in world coordinates,
  # the other for the shape in image coordinates
  fig = P.figure(0);
  fig.clear()
  ax1 = fig.add_subplot(1,2,1)
  ax2 = fig.add_subplot(1,2,2)

  # the shape in world coordinates
  s.draw(ax1)
  ax2.axis('scaled')
  ax1.axis([min(xy[:,0])-1, max(xy[:,0])+1, min(xy[:,1])-1, max(xy[:,1])+1])
  ax1.set_title('World coordinates')

  # the shape in image coordinates
  A = matrix_rigid3d(cam)
  u, v = transform_point(xy[:,0], xy[:,1], zeros(len(xy)), A)
  s.draw(ax2, zip(u,v))
  ax2.set_title('Image coordinates')
  ax2.axis('scaled')
  ax2.axis([min(u), max(u), min(v), max(v)])

  P.draw()


def pnp_experiments(Shape, n=1, observation_noise=[1.],
                    initialization_noise=[1.]):
  """Under a fixed camera's pose, render the given shape, then perturb
  the observed image coordinates with different amounts of noise, and
  run a calibration procedure. Report the error of the calibration
  procedure. also do this for different amounts of noise in the
  initial iterate of the calibration procedure.
  """
  # ground truth camera pose
  cam=(1.,0.,0.,  0.,-1.4,4., 300.)


  # project shape to image coordinates
  s = Shape()
  xy = array(s.endpoints())
  A = matrix_rigid3d(cam)
  u, v = transform_point(xy[:,0], xy[:,1], zeros(len(xy)), A)


  df = {}

  for init_noise in initialization_noise:
    for obs_noise in observation_noise:
      for it in xrange(n):
        # corrupt image observation by gaussian noise
        noise_var = obs_noise/sqrt( (max(u)-min(u) + max(v) - min(v))/2 )
        u += noise_var*random.randn(len(u))
        v += noise_var*random.randn(len(v))

        # perturb starting pose slightly
        cam_init = cam + hstack((init_noise*0.1*random.randn(3),
                                 init_noise*random.randn(3),
                                 init_noise*10*random.randn(1) ))

        # recover pose
        cam_hat = PnP(xy[:,0], xy[:,1], zeros(len(xy)),
                      u,v,cam_init,(0,0), w=1., rendering=False)

        # rotation error
        df.setdefault('noise_var',[]).append(noise_var)
        df.setdefault('init_noise',[]).append(init_noise)
        df.setdefault('rot_err',[]).append(nor(cam_hat[:3] - cam[:3]))
        df.setdefault('translation_err',[]).append(nor(cam_hat[3:6] - cam[3:6]))
        df.setdefault('f_err',[]).append(abs(cam_hat[6] - cam[6]))

  for k,v in df.iteritems():
    df[k] = array(v)

  return df




def show_pnp_experiments(df):
  """Renter the output of pnp_experiments"""

  fig = P.figure(0); fig.clear()
  ax_rot = fig.add_subplot(1,3,1)
  ax_trans = fig.add_subplot(1,3,2)
  ax_f = fig.add_subplot(1,3,3)

  colors = ['r','g','b','k','m','c']
  init_noises = unique(df['init_noise'])

  for init_noise,color in zip(init_noises,colors):
    i = df['init_noise'] == init_noise

    ax_rot.plot(df['noise_var'][i], df['rot_err'][i],
                color=color, lw=0, marker='.')
    ax_trans.plot(df['noise_var'][i], df['translation_err'][i],
                  color=color, lw=0, marker='.')
    ax_f.plot(df['noise_var'][i], df['f_err'][i],
             color=color, lw=0, marker='.')

  ax_rot.set_ylim([0,1.])
  ax_trans.set_ylim([0,2.])
  ax_f.set_ylim([0,80.])
  ax_rot.set_title('Rotation error')
  ax_trans.set_title('Translation error')
  ax_trans.set_xlabel('Relative image noise variance')
  ax_f.set_title('Focal length error')
  ax_f.legend(['Initialization noise=%.4g'%n for n in init_noises])


from ggplot import *
def show_pnp_experiments0(df):
  dfm = pd.melt(df, id_vars=['init_noise', 'noise_var'],
                var_name='parameter', value_name='err')
  return ggplot(dfm, aes(x='noise_var', y='err', color='init_noise')) + stat_smooth() + geom_point() + facet_wrap(x='parameter')

def transform_point(x,y,z,A,dAdp=None):
    """Apply a 3x4 linear transform A to points in x,y,z and project
    the resulting points back.  If dA/dp is specified, also returns
    the derivatives of the points wrt dp. Each entry of dAdp is the
    Jacobian of A wrt some variable p.

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


def PnP(x,y,z,u,v,cam,center, w=1., rendering=True):
    """Finds camera parameters (pose and focal length) that minimize
    the projection error between real-world points X in the XY
    plane at Z=1 and 2D image points uv. An initial pose must be
    supplied.
    """
    wsqrt = sqrt(w)

    def _func(a):
        A,dA = matrix_rigid3d(a,deriv=True)
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
