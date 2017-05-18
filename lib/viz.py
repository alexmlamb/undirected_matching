"""
Tools for plotting / visualization
"""

import sys
import warnings

import imageio
import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# UTILS ########################################################################


def is_square(shp, n_colors=1):
    """
    Test whether entries in shp are square numbers, or are square numbers after divigind out the
    number of color channels.
    """
    is_sqr = (shp == np.round(np.sqrt(shp))**2)
    is_sqr_colors = (shp == n_colors*np.round(np.sqrt(np.array(shp)/float(n_colors)))**2)
    return is_sqr | is_sqr_colors

def show_receptive_fields(theta, P=None, n_colors=None, max_display=100,
                          grid_wa=None, labels=None, captions=None, title=""):
    """
    Display receptive fields in a grid. Tries to intelligently guess whether to treat the rows,
    the columns, or the last two axes together as containing the receptive fields. It does this
    by checking which axes are square numbers -- so you can get some unexpected plots if the wrong
    axis is a square number, or if multiple axes are. It also tries to handle the last axis
    containing color channels correctly.
    """

    shp = np.array(theta.shape)
    if n_colors is None:
        n_colors = 1
        if shp[-1] == 3:
            n_colors = 3
    # multiply colors in as appropriate
    if shp[-1] == n_colors:
        shp[-2] *= n_colors
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge last two axes
        shp[-2] *= shp[-1]
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge leading axes
        theta = theta.reshape((-1,shp[-1]))
        shp = np.array(theta.shape)
    if len(shp) == 1:
        theta = theta.reshape((-1,1))
        shp = np.array(theta.shape)

    # figure out the right orientation, by looking for the axis with a square
    # number of entries, up to number of colors. transpose if required
    is_sqr = is_square(shp, n_colors=n_colors)
    if is_sqr[0] and is_sqr[1]:
        warnings.warn("Unsure of correct matrix orientation. "
            "Assuming receptive fields along first dimension.")
    elif is_sqr[1]:
        theta = theta.T
    elif not is_sqr[0] and not is_sqr[1]:
        # neither direction corresponds well to an image
        # NOTE if you delete this next line, the code will work. The rfs just won't look very
        # image like
        return False

    theta = theta[:,:max_display].copy()

    if P is None:
        img_w = int(np.ceil(np.sqrt(theta.shape[0]/float(n_colors))))
    else:
        img_w = int(np.ceil(np.sqrt(P.shape[0]/float(n_colors))))
    nf = theta.shape[1]
    if grid_wa is None:
        grid_wa = int(np.ceil(np.sqrt(float(nf))))
    grid_wb = int(np.ceil(nf / float(grid_wa)))
    
    if captions is not None:
        grid_wa *= 2

    if P is not None:
        theta = np.dot(P, theta)

    vmin = np.min(theta)
    vmax = np.max(theta)

    for jj in range(nf):
        if captions is not None:
            jj_ = 2 * grid_wb * (jj // grid_wb) + (jj % grid_wb)
        else:
            jj_ = jj
            
        plt.subplot(grid_wa, grid_wb, jj_ + 1)
        
        if jj == int(np.sqrt(nf)/2) - 1:
            plt.title(title)

        ptch = np.zeros((n_colors * img_w ** 2,))
        ptch[:theta.shape[0]] = theta[:,jj]
        if n_colors==3:
            ptch = ptch.reshape((n_colors, img_w, img_w))
            ptch = ptch.transpose((1, 2, 0)) # move color channels to end
        else:
            ptch = ptch.reshape((img_w, img_w))
        #ptch -= vmin
        #ptch /= vmax-vmin
        
        plt.imshow(ptch, interpolation='nearest', cmap=cm.Greys_r, vmin = 0.0,
                   vmax = 1.0)
        
        if labels is not None:
            if len(str(labels[jj])) > 8:
                plt.text(0., -1., labels[jj], fontsize=6)
            elif len(str(labels[jj])) > 16:
                plt.text(0., -1., labels[jj], fontsize=4)
            elif len(str(labels[jj])) > 24:
                plt.text(0., -1., labels[jj], fontsize=2)
            else:
                plt.text(0., -1., labels[jj])
                
        plt.axis('off')
        if captions is not None:
            plt.subplot(grid_wa, grid_wb, jj_ + 1 + grid_wb)
            cap = []
            for cap_ in captions[jj]:
                if len(cap_) < 20:
                    cap.append(cap_)
                else:
                    cap += [cap_[x:x+20] for x in range(0, len(cap_), 20)]
            plt.text(0., 1., '\n'.join(cap), fontsize=8,
                     verticalalignment='top')

        plt.axis('off')

    return True


def plot_images(X=None, labels=None, captions=None, title="",
                file_path=None):
    Xcol = X.reshape((X.shape[0], -1,)).T
    if captions is None:
        plt.figure(figsize=[8, 8])
    else:
        plt.figure(figsize=[16, 32])
    
    if show_receptive_fields(Xcol, n_colors=X.shape[1], labels=labels,
                             title=title, captions=captions):
        plt.savefig(file_path + '.png')
    else:
        warnings.warn('Images unexpected shape.')
    
    plt.close()
    
   
def plot_chain(chain, shape=(8, 8), dim_c=None, dim_x=None, dim_y=None,
               file_path=None):
    chain = chain[:, :(shape[0]*shape[1]), ...]
    chain_ = []
    for i in xrange(chain.shape[0]):
        x = chain[i]
        x = x.reshape(shape[0], shape[1], dim_c, dim_x, dim_y)
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape(shape[0] * dim_x, shape[1] * dim_y, dim_c)
        chain_.append(x)
    imageio.mimsave(file_path, chain_)
    
    
def plot_inpaint_chain(chain, x_gt, shape=(8, 8), dim_c=None, dim_x=None,
                       dim_y=None, file_path=None):
    chain_ = []
    for x in chain:
        x = x.reshape(shape[0], shape[1], dim_c, dim_x, dim_y)
        x_ = np.zeros((2 * shape[0], shape[1], dim_c, dim_x, dim_y))
        x_[:shape[0]] = x
        x_[shape[0]:] = x_gt[:shape[0]*shape[1]].reshape(
            shape[0], shape[1], dim_c, dim_x, dim_y)
        x = x_.transpose(0, 3, 1, 4, 2)
        x = x.reshape(2 * shape[0] * dim_x, shape[1] * dim_y, dim_c)
        chain_.append(x)
    imageio.mimsave(file_path, chain_)


if __name__ == "__main__":

    import numpy.random as rng

    x = rng.normal(size = (64,1,28,28))

    plot_images(x, fname = "derp.png", title = "DERP DERP DERP")

