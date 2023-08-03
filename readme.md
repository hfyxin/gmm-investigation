# Random Investigation on Clusterting

## GMM (or Gaussian distribution)

Run test_gmm.py to see the results.

### How to determine if a point X falls within a GMM cluster

#### Method 1: calculate CDF

Given the Probability density function (PDF), one can calculate Cumulative probability function (CDF), which looks like an S curve in 1D setting. This gives the probability value between 0 and 1, representing a probability of X belongs to a GMM.

- But one problem is CDF is cannot be expressed in terms of elementary functions, instead in Math it introduces an error function erf(x). Numerical method or lookup table is required to calculate this function. Nevertheless MATLAB and SciPy provides functions to calculate them.
- Second problem is that CDF is 2D or more is ill-defined. It simply does integral from (-inf, -inf) to point (x, y), which does not represent a probability. See below side by side of a bivariate Gaussian pdf and cdf:

A bivariate Gaussian pdf (left) and cdf (right)

![A bivariate Gaussian pdf (left) and cdf (right)](https://github.com/hfyxin/gmm-investigation/blob/main/images/pdf-vs-cdf.png?raw=true)


#### Method 2: calculate PDF and put a threshold

Yes and simple to implement. But the problem is pdf(x, y) is dependent on sigma (covariance). If two distribution’s covariance are different, their pdf values are not comparable. A pdf threshold is pointless.

Alternatively one can determine the pdf threshold by looking at its n-sigma values. e.g. calculating pdf(n-sigma). This is easy to do in 1D case, or 2D if the covariance is diagonal. But in 2D case if the distribution ellipse is not parallel to x or y axis, 1-sigma values (x,y) is difficult to obtain. the angle of rotation is required, which makes the Method 3 actually more convenient.

### Method 3: Calculate n-sigma thresholds as ellipse/ellipsoid

This is similar to Method 2 but instead specifying n-sigma as thresholds; n can be any positive number.

The general thought process is:

- if n=1, simply determine if mu-sigma < x < mu+sigma
- if n=2, calculate the oval equation, decide if X falls within the oval.
- if n=3, use ellipsoid equation.
- if n>=4, use high-dimensional ellipsoid equation.

The main challenge is to deal with rotation in n=2,3,4.. cases.

**TL, DR. The rotation is defined by the eigenvectors of covariance matrix.** Eigenvectors consist a matrix R. R defines the rotation of the oval which is originally at origin. Radii of oval is defined by eigenvalues.

For a new point X(x1, x2, x3, ..), simply do R' * (X-mu), plug it into oval equation.

#### The Long Story

**Translation and rotation in 2D case (ellipse)**

Steps to draw (imagine) an ellipse:

- determine its two radii, a and b, draw upright ellipse on origin. 
- rotate ccw by theta degrees.
- translate to a new center location mu
- (a,b, theta all calculated from GMM covariance)

Steps to determine if a point X falls within such ellipse (reverse the process):

- translate X by -mu. 
- rotate cw by theta.
- apply ellipse equation.

**Translation and rotation in 3D case (ellipsoid)**

Similarly draw an ellipsoid in 3D at origin. its orientation is defined by the unit vector diag([1,1,1]). Then it rotates to the orientation defined by covariance eigenvectors.

- One caveat is that the rotation is with regard to a set of three vectors rather than just x direction. It is no problem to use just x direction in 2D, but in 3D the rotation will not be unique.
  - I investigated quaternion rotation based on the assumption that rotation around x is good enough. But it failed obviously.
- Another problem is that the eigenvectors, being orthonormal vectors, can be either right-handed or left-handed and the computation result can be unpredictable. It leads to errors.
  - Calculating vector set rotation can use the Orthogonal Procrustes problem. Pay attention to the handedness of the result.
  - Since the original position is at origin, the best way is to just use eigenvectors directly as rotation matrix, regardless of left or right handed.


## Design Validation

Everything done in Python. Use pdf method (Method 2) above and apply a threshold. The threshold won’t match exactly with the n-sigma method. Therefore compare only the shape and position of ellipse and ellipsoid.

### 2D Bi-variate Normal Distribution

- Generate a grid of data points (x, y) → use n-sigma method to highlight points within, and draw ellipse
- Pdf method → use Scipy function to draw surface whose color represents magnitudes
- Vary mean and covariance to create test cases.

Verifying bivariate distribution
![Verifying bivariate distribution](https://github.com/hfyxin/gmm-investigation/blob/main/images/verify-1.png?raw=true)

Caveat, when covariance is not full-rank, eigenvectors cannot be calculated. This case may not exist.

### 3D tri-variate Normal Distribution

- Generate a grid of data points (x, y, z) → use n-sigma method to select points within, and scatter3d
- Pdf method → use Scipy function to calculate values and apply a (rough) threshold
- Vary mean and covariance to create test cases.


Verifying tri-variate (3D) distribution. Note the shape and location are same, although threshold is different

![Verifying tri-variate (3D) distribution](https://github.com/hfyxin/gmm-investigation/blob/main/images/verify-2.png?raw=true)

Verifying tri-variate (3D) distribution
![Verifying tri-variate (3D) distribution 2](https://github.com/hfyxin/gmm-investigation/blob/main/images/verify-3.png?raw=true)

