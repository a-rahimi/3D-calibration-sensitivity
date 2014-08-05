import pdb
import pylab as P
from numpy import *
import pandas as pd
from calibration import PnP, matrix_rigid3d, transform_point

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
    """
            C
            /\
           /  \
          B G  D
         /      \
        /        \
       A    F     E
    """
    return (-2,0), (-1,3), (0,6), (1,3), (2,0), (0,0), (0,3)

  def segments(self, xy):
    a,b,c,d,e,f,g = xy
    return [hstack((a,b)), hstack((b,c)), hstack((c,d)), hstack((d,e)),
            hstack((f,g)), hstack((g,c))]

class ShapeGrid(Shape):
  nx = 3
  ny = 3

  def endpoints(self):
    X,Y = meshgrid(linspace(-2.,2., self.nx), linspace(0.,6., self.ny))
    return zip(X.ravel(), Y.ravel())

  def segments(self, xy):
    X,Y = array(xy).T
    X = X.reshape(self.ny, self.nx)
    Y = Y.reshape(self.ny, self.nx)
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


def pnp_experiments(Shape, cam, n=1, observation_noise=[1.],
                    initialization_noise=[1.]):
  """Under a fixed camera's pose, render the given shape, then perturb
  the observed image coordinates with different amounts of noise, and
  run a calibration procedure. Report the error of the calibration
  procedure. also do this for different amounts of noise in the
  initial iterate of the calibration procedure.
  """
  # shape in world coordinates
  s = Shape()
  xy = array(s.endpoints())

  # project shape to image coordinates
  A = matrix_rigid3d(cam)
  u, v = transform_point(xy[:,0], xy[:,1], zeros(len(xy)), A)

  df = {}
  for init_noise in initialization_noise:
    for obs_noise in observation_noise:
      for it in xrange(n):
        # corrupt image observation by gaussian noise
        noise_var = obs_noise/sqrt( (max(u)-min(u) + max(v) - min(v))/2 )
        uo = u + noise_var*random.randn(len(u))
        vo = v + noise_var*random.randn(len(v))

        # perturb starting pose slightly
        cam_init = cam + hstack((init_noise*0.1*random.randn(3),
                                 init_noise*random.randn(3),
                                 init_noise*10*random.randn(1) ))

        # recover pose
        cam_hat = PnP(xy[:,0], xy[:,1], zeros(len(xy)),
                      uo, vo, cam_init, (0,0), w=1., rendering=False)

        # rotation error
        df.setdefault('noise_var',[]).append(noise_var)
        df.setdefault('init_noise',[]).append(init_noise)
        df.setdefault('rot_err',[]).append(nor(cam_hat[:3] - cam[:3]))
        df.setdefault('translation_err',[]).append(nor(cam_hat[3:6] - cam[3:6]))
        df.setdefault('f_err',[]).append(abs(cam_hat[6] - cam[6]))

  for k,v in df.iteritems():
    df[k] = array(v)

  return df




def show_pnp_experiments(df, ylim_factor=1.):
  """Renter the output of pnp_experiments"""

  # one set of axes for each of rotation, translation and focal point.
  fig = P.figure(0); fig.clear()
  ax_rot = fig.add_subplot(1,3,1)
  ax_trans = fig.add_subplot(1,3,2)
  ax_f = fig.add_subplot(1,3,3)

  colors = ['r','g','b','k','m','c']
  init_noises = unique(df['init_noise'])


  # draw the curves for a given constant initialization in each of the
  # axes
  for init_noise,color in zip(init_noises,colors):
    i = df['init_noise'] == init_noise

    x_dodge = .1* mean(abs(diff(unique(df['noise_var'][i])))) * random.randn(sum(i))

    ax_rot.plot(df['noise_var'][i] + x_dodge,
                df['rot_err'][i],
                color=color, lw=0, marker='.', ms=1)
    ax_trans.plot(df['noise_var'][i] + x_dodge,
                  df['translation_err'][i],
                  color=color, lw=0, marker='.', ms=1)
    ax_f.plot(df['noise_var'][i] + x_dodge,
              df['f_err'][i],
              color=color, lw=0, marker='.', ms=1)

  ax_rot.set_ylim([0,ylim_factor*1.])
  ax_trans.set_ylim([0,ylim_factor*2.])
  ax_f.set_ylim([0,ylim_factor*80.])
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
