---
layout: default
title:  "Linear Alegebra and Interpolation"
date:   2022-09-16 21:22:34 +0900
usemathjax: true
# nav_order : 1
# collection: 
# parent: 예전 자료
has_toc: true
permalink: /la
nav_exclude: true
---

# Linear Algebra with SciPy
The main Python package for linear algebra is the SciPy subpackage `scipy.linalg` which builds on NumPy. Let's import both packages:

#### Inverse
We can find the inverse using the funciton `scipy.linalg.inv`:
### scipy.linalg.eig
The funciton `scipy.linalg.eig` computes eigenvalues and eigenvectors of a square matrix $A$.
Let's consider a simple example with a diagonal matrix:

# Interpolation
## Formulation
Suppose we have $n+1$ points in the $xy$-plane $$(x_{0},y_{0}), (x_{1}, y_{1}), \dots, (x_{n},y_{n})$$ such that all the $x$ values are distinct
$(x_{i} \neq x_{j} \text{  for  } i\neq j)$. The general form of a degree $n$ polynomial is $$p(x)=a_{0} + a_{1}x+a_{2}x^{2} + \dots + a_{n}x^{n}$$
If $p(x)$ is the unique degree $n$ polynomial which interpolates all the points, then the coefficeints $a_{0}, a_{1}, \dots, a_{n}$ satisfy the following equations:
$$\begin{align*}
a_{0} + a_{1}x_{0} + a_{2}x_{0}^{2} + \dots + a_{n}x_{0}^{n} &= y_{0}\\
a_{0} + a_{1}x_{1} + a_{2}x_{1}^{2} + \dots + a_{n}x_{1}^{n} &= y_{1}\\
&\vdots\\
a_{0} + a_{1}x_{n} + a_{2}x_{n}^{2} + \dots + a_{n}x_{n}^{n} &= y_{n}
\end{align*}$$
Therefore the vector of coefficents
$$ a = \begin{bmatrix} a_{0} \\ a_{1} \\ \vdots \\ a_{n} \end{bmatrix}$$
is the unique the solution of the linear system of equations $$Xa=y$$
where $X$ is the Vandermonde matrix and $y$ is the vector of $y$ values 
$$ X =
\begin{bmatrix}
1 & x_{0} & x_{0}^{2} & \dots &x_{0}^{n}\\
1 & x_{1} & x_{1}^{2} & \dots &x_{1}^{n}\\
  & \vdots &           &       &\vdots \\
1 & x_{n} & x_{n}^{2} & \dots &x_{n}^{n}
\end{bmatrix}
\quad \text{and} \quad y = \begin{bmatrix}
y_{0} \\
y_{1} \\
y_{2} \\
\vdots \\
y_{n} \end{bmatrix}$$


## Least Sqaures Linear Regression
Suppose we have $n+1$ points $$(x_{0}, y_{0}), (x_{1},y_{1}), \dots, (x_{n},y_{n})$$ in the $xy$-plane and we want to fit a line 
$$y=a_{0} +a_{1}x$$
that "best fits" the data. There are different ways to quantify what "best fit" means but the most common method is called least squares linear regression. In least squares linear regression, we want to minimize the sum of squared errors
$$SSE = \sum_{i}(y_{i}-(a_{0}+a_{i}x_{i}))^{2}$$
### Formulation
If we form matrices 
$$ X = 
\begin{bmatrix} 
1 & x_{0}\\
1 & x_{1}\\
\vdots & \vdots\\
1 & x_{n}
\end{bmatrix},
\mathbf{y} = \begin{bmatrix}
y_{0}\\
y_{1}\\
\vdots\\
y_{n}
\end{bmatrix}

\mathbb{a} = \begin{bmatrix} a_{0} \\a_{1} \end{bmatrix}$$
then the sum of squared errors can be expressed as $$SSE = \left|\mathbf{y}-X\mathbf{a}\right|^{2}$$

**Theorem.** (Least Squares Regression) Consider $n+1$ points
$$\left(x_{0}, y_{0}\right), \left(x_{1},y_{1}\right), \dots, \left(x_{n}, y_{n}\right)$$
in the $xy$-plane. The coefficients $\mathbf{a}=\left[a_{0}, a_{1} \right]^{T}$ which minimize the sum of squared errors
$$SSE = \sum_{i} (y_{i}-(a_{0}+a_{i}x_{i}))^{2}$$
is the unique solution of the system
$$(X^{T}X)\mathbf{a} = X^{T}\mathbf{y}$$

*Sketch of Proof.* The product $X\mathbf{a}$ is in the column space of $X$. The line connecting $y$ to the nearest point in the column space of $X$ is perpendicular to the column space to $X$. Therefore 
$$X^{T}(\mathbf{y}-X\mathbf{a}) = \mathbf{0}$$ 
and so
$$(X^{T}X)\mathbf{a} = X^{T}\mathbf{y}$$