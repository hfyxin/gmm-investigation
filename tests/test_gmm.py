import sys
sys.path.append('./')
from gmm import x_fall_within_sigma_1d, x_fall_within_sigma_2d, x_fall_within_sigma_3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def test_x_fall_within_sigma_1d():
    n = 1
    mean = np.array([0])
    cov = np.array([[1]])
    test_cases = np.arange(-10,10,0.5)
    n_sigma = 1
    for x in test_cases:
        print(f"{x=} {n_sigma=} {x_fall_within_sigma_1d(x, mean, cov, n_sigma)}")


def test_x_fall_within_sigma_2d():
    # generate a grid of points
    x1 = np.arange(-5, 5, 0.5)
    x2 = np.arange(-5, 5, 0.5)
    xx1, xx2 = np.meshgrid(x1, x2)
    test_cases = np.c_[xx1.ravel(), xx2.ravel()]
    print(f"{test_cases.shape=}")

    # define a 2d Gaussian distribution
    n = 2 
    mean = np.array([1,0])
    cov = np.array([[1, 1.732], [1.732, 4]])

    print(f"Testing x fall within {1}-sigma ellipse. mean={mean}, cov={cov}")
    print(f"|  x1   |   x2   | within 1-sigma |")
    print(f"|-------|--------|----------------|")
    for x in test_cases:
        print(f"| {x[0]:.2f} | {x[1]:.2f} | {x_fall_within_sigma_2d(x, mean, cov, n_sigma=1)} |")
        

def test_x_fall_within_sigma_2d_visual():

    # define a 2d Gaussian distribution
    n = 2 
    mean = np.array([1,0])
    cov = np.array([[1, 1.732], [1.732, 4]])
    mean_cases = []
    mean_cases.append(np.array([1,0]))
    mean_cases.append(np.array([0,1]))
    mean_cases.append(np.array([0,0]))
    mean_cases.append(np.array([1,1]))
    mean_cases.append(np.array([-1,-1]))
    cov_cases = []
    cov_cases.append(np.array([[1, 1.732], [1.732, 4]]))
    cov_cases.append(np.array([[1, 0], [0, 1]]))
    cov_cases.append(np.array([[1, -np.sqrt(0.5)], [-np.sqrt(0.5), 1]]))
    cov_cases.append(np.array([[1, 0], [0, 4]]))
    cov_cases.append(np.array([[1, -0.5], [-0.5, 1]]))
    cov_cases.append(np.array([[1, 1], [1, 1]]))

    
    print(f"Testing guassian ellipse shape")
    for mean in mean_cases:
        for cov in cov_cases:
            print(f"mean={mean}, cov={cov}")
            show_2d_gaussian_plots(mean, cov)


def show_2d_gaussian_plots(mean, cov):
    # generate a grid of points
    x1 = np.arange(-5, 5, 0.5)
    x2 = np.arange(-5, 5, 0.5)
    xx1, xx2 = np.meshgrid(x1, x2)
    test_cases = np.c_[xx1.ravel(), xx2.ravel()]
    print(f"{test_cases.shape=}")

    # calculate the points that fall within 1-sigma
    x_in_oval = []  # points that fall within sigma
    x_out_oval = []
    for i in range(test_cases.shape[0]):
        y = x_fall_within_sigma_2d(test_cases[i,:], mean, cov, n_sigma=1)
        if y:
            x_in_oval.append(test_cases[i,:])
        else:
            x_out_oval.append(test_cases[i,:])
    x_in_oval = np.array(x_in_oval)
    x_out_oval = np.array(x_out_oval)

    # ellipse values
    eigv, eigw = np.linalg.eigh(cov)
    std = np.sqrt(eigv)     # major and minor radius of ellipse
    angle = np.arctan2(eigw[1][0], eigw[0][0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    print(f"{mean=}, {eigv=}, {std=}, {eigw=}, {angle=:.2f}")

    # set up the plot
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1)
    ax.set_title("ellipsoid method")
    if x_in_oval.shape[0] != 0:
        ax.scatter(x_in_oval[:,0], x_in_oval[:,1], c="b")
    if x_out_oval.shape[0] != 0:
        ax.scatter(x_out_oval[:,0], x_out_oval[:,1], c="r")
    ellipse = Ellipse(mean, 2 * std[0], 2 * std[1], angle=angle)
    ellipse.set_clip_box(fig.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)
    ax.grid()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # plot gaussian distribution as a heatmap
    ax = fig.add_subplot(1,2,2)
    ax.set_title("pdf method")
    from scipy.stats import multivariate_normal
    try:
        y = multivariate_normal.pdf(test_cases, mean, cov)
        y = y.reshape(xx1.shape)
        ax.contourf(xx1, xx2, y)
    except Exception as e:
        print(f"{e=}")
    ax.grid()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.show()

def test_x_fall_within_sigma_3d():
    n=3
    mean = np.array([0,0,0])
    cov = np.array([[9,0,0],[0,1,0],[0,0,1]])
    print(x_fall_within_sigma_3d(np.array([0,0,0]), mean, cov, n_sigma=1))


def test_x_fall_within_sigma_3d_visual():
    # generate a 3d grid of points
    x_range = (-10, 10)
    y_range = (-5, 5)
    z_range = (-8, 8)
    x1 = np.arange(*x_range, 0.5)
    x2 = np.arange(*y_range, 0.5)
    x3 = np.arange(*z_range, 0.5)
    xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
    X = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]
    print(f"{X.shape=}")

    n=3
    mean = np.array([0,0,0])
    std = np.array([1,2,3])
    eigw = np.array([[1,0,0],[0,-np.sqrt(0.5),-np.sqrt(0.5)],[0,-np.sqrt(0.5),np.sqrt(0.5)]])  # this is a left hand eigenvector
    std_cases = generate_std_cases()
    eigw_cases = generate_eigw_cases()


    for k, eigw in enumerate(eigw_cases):
        n_cases = len(eigw_cases)
        cov = np.dot(np.dot(eigw, np.diag(std * std)), np.linalg.inv(eigw))
        print(f"Test case {k+1}: \n{mean=}, {std=},\ncov=\n{cov}\neigw=\n{eigw}\n")
        
        print(f"Ellipsoid method:")
        x_plot = []  # points that fall within sigma
        x_fall_within_sigma_3d(X[1,:], mean, cov, n_sigma=2, verbose=True)
        for i in range(X.shape[0]):
            y = x_fall_within_sigma_3d(X[i,:], mean, cov, n_sigma=2)
            if y:
                x_plot.append(X[i,:])

        x_plot = np.array(x_plot)            
        print(f"{x_plot.shape=}")

        # set up the ellipsoid method plot
        fig = plt.figure(0)  # Square figure
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f"3D Gaussian distribution #{k+1}/{n_cases} (ellipsoid method)")
        if x_plot.shape[0] != 0:
            # ax.scatter(X[:,0], X[:,1], X[:,2])
            ax.scatter(x_plot[:,0], x_plot[:,1], x_plot[:,2])
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            ax.set_xlim(*x_range)
            ax.set_ylim(*y_range)
            ax.set_zlim(*z_range)
            ax.axis('equal')
        

        # scipy method
        from scipy.stats import multivariate_normal
        y = multivariate_normal.pdf(X, mean, cov)
        pdfThres = 0.0031
        x_plot = X[y > pdfThres, :]
        print(f"pdf method: {x_plot.shape=}")

        # set up another plot
        fig1 = plt.figure(1)  # Square figure
        ax = fig1.add_subplot(projection='3d')
        ax.set_title(f"3D Gaussian distribution #{k+1}/{n_cases} (pdf method)")
        ax.scatter(x_plot[:,0], x_plot[:,1], x_plot[:,2])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        ax.axis('equal')
        plt.show()


def do_test_nbr_4():
    # generate a 3d grid of points
    x_range = (-10, 10)
    y_range = (-5, 5)
    z_range = (-8, 8)
    x1 = np.arange(*x_range, 0.5)
    x2 = np.arange(*y_range, 0.5)
    x3 = np.arange(*z_range, 0.5)
    xx1, xx2, xx3 = np.meshgrid(x1, x2, x3)
    X = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]
    print(f"{X.shape=}")
    
    mean = np.array([0,0,0])
    std = np.array([1,2,3])
    eigw = np.array([[0,1,0],[0,0,1],[1,0,0]])
    cov = np.array([[4., 0., 0.], 
                    [0., 9., 0.], 
                    [0., 0., 1.]])
    print(f"Ellipsoid method:")
    x_plot = []  # points that fall within sigma
    x_fall_within_sigma_3d(X[1,:], mean, cov, n_sigma=2, verbose=True)
    for i in range(X.shape[0]):
        y = x_fall_within_sigma_3d(X[i,:], mean, cov, n_sigma=2)
        if y:
            x_plot.append(X[i,:])

    x_plot = np.array(x_plot)            
    print(f"{x_plot.shape=}")

    # set up the ellipsoid method plot
    fig = plt.figure(0)  # Square figure
    ax = fig.add_subplot(projection='3d')
    ax.set_title(f"3D Gaussian distribution (ellipsoid method)")
    if x_plot.shape[0] != 0:
        # ax.scatter(X[:,0], X[:,1], X[:,2])
        ax.scatter(x_plot[:,0], x_plot[:,1], x_plot[:,2])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        ax.axis('equal')
    

    # scipy method
    from scipy.stats import multivariate_normal
    y = multivariate_normal.pdf(X, mean, cov)
    pdfThres = 0.0031
    x_plot = X[y > pdfThres, :]
    print(f"pdf method: {x_plot.shape=}")

    # set up another plot
    fig1 = plt.figure(1)  # Square figure
    ax = fig1.add_subplot(projection='3d')
    ax.set_title(f"3D Gaussian distribution (pdf method)")
    ax.scatter(x_plot[:,0], x_plot[:,1], x_plot[:,2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)
    ax.axis('equal')
    plt.show()



def compare_real_vs_rotation():
    # generate cov matrix
    n=3
    mean = np.array([0,0,0])
    std = np.array([3,2,1])
    std_cases = generate_std_cases()
    
    eigw = np.array([[1,0,0], [0,1,0],[0,0,1]])
    eigw_cases = generate_eigw_cases()
    for i, eigw in enumerate(eigw_cases):
        cov = np.dot(np.dot(eigw, np.diag(std * std)), np.linalg.inv(eigw))
        print("-----------------------------------------------")
        print(f"Test case {i}: {std=},\n{cov=},\n{eigw=}\n")
        # just to print out values
        x_fall_within_sigma_3d(np.array([1,1,1]), mean, cov, n_sigma=2, verbose=True)


def generate_std_cases():
    std_cases = []
    std_cases.append(np.array([3,2,1]))
    std_cases.append(np.array([3,1,2]))
    std_cases.append(np.array([2,3,1]))
    std_cases.append(np.array([2,1,3]))
    std_cases.append(np.array([1,3,2]))
    std_cases.append(np.array([1,2,3]))
    return std_cases

def generate_eigw_cases():
    eigw_cases = []
    eigw_cases.append(np.array([[1,0,0], [0,1,0],[0,0,1]]))
    eigw_cases.append(np.array([[1,0,0], [0,0,1],[0,1,0]]))
    eigw_cases.append(np.array([[0,1,0], [1,0,0],[0,0,1]]))
    eigw_cases.append(np.array([[0,1,0], [0,0,1],[1,0,0]]))
    eigw_cases.append(np.array([[0,0,1], [1,0,0],[0,1,0]]))
    eigw_cases.append(np.array([[0,0,1], [0,1,0],[1,0,0]]))
    return eigw_cases


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    test_x_fall_within_sigma_1d()
    test_x_fall_within_sigma_2d()
    test_x_fall_within_sigma_2d_visual()
    test_x_fall_within_sigma_3d()
    test_x_fall_within_sigma_3d_visual()
    do_test_nbr_4()
    compare_real_vs_rotation()