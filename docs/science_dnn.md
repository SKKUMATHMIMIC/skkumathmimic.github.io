---
layout: default
title:  "Basic"
date:   2022-09-16 21:22:34 +0900
usemathjax: true
# nav_order : 1
# collection: 
# parent: 예전 자료
has_toc: true
nav_exclude: true
permalink: /basic
---
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# numpy linear algebra matrix multiplication
Matrix powers are performed by the function `numpy.linalg.matrix_power`. It's a long function name and so it's convenient to import it with a shorter name:


# Random Number Generators
The subpackage `numpy.random` contains functions to generate NumPy arrays of random numbers sampled from different distributions. The following is a partial list of distributions:
|Funcions|Description|
|:---|:---|
|`numpy.random.rand(d1, ..., dn)`|Create a NumPy array (with shape (d1, ..., dn)) with entries sampled uniformly from (0, 1)|
|`numpy.random.randn(d1, ..., dn)`|Create a NumPy array (with shape (d1, ..., dn)) with entries sampled from the standard normal distribution|
|`numpy.random.randint(a, b, size)`|Create a NumPy array (with shape size) with integer entries from low (inclusive) to high (exclusive)

Sample a random number from the uniform distribution:

# Bisection Error
## Absolute Error 

The bisection method does not (in general) produce an exact solution of an equation $f(x) = 0$. However, we can give an estimate of the absolute error in the approximation.
* * *
**Theorem**. Let $f(x)$ be a continuous function on $[a,b]$ such that $f(a)f(b)<0$. After $N$ iterations of the bisection method, let $x_{N}$ be the midpoint in the $N$th subinterval $[a_{N},b_{N}]$ $$x_{N}=\frac{a_{N}+b_{N}}{2}$$ There exists an exact solution $x_{\text{true}}$ of the equation $f(x)=0$ in the subinterval $[a_{N},b_{N}]$ and the absolute error is $$ \|x_{\text{true}-x_{N}}\| \leq \frac{b-a}{2^{N+1}}$$
- - -
Note that we can rearrange the error bound to see the minimum number of iterations required to guarantee absolute error less than a prescribed $\epsilon$:
$$\frac{b-a}{2^{N+1}} < \epsilon$$ 
$$\frac{b-a}{\epsilon} < 2^{N+1}$$
$$\ln \left(\frac{b-a}{\epsilon}\right) < (N+1)\ln(2)$$
$$\frac{\ln\left(\frac{b-a}{\epsilon}\right)}{\ln(2)} -1 < N$$ 


# Secant
## Algorithm
The secant method procedure is almost identical to the bisection method. The only difference it how we divide each subinterval.
1. Choose a starting interval $[a_{0},b_{0}]$ such that $f(a_{0})f(b_{0})<0$.
1. Compute $f(x_{0})$ where $X_{0}$ is given by the secant line $$ x_{0} = a_{0} -f(a_{0})\frac{b_{0}-a_{0}}{f(b_{0})-f(a_{0})}$$
1. Determine the next subinterval $[a_{1},b_{1}]$:
    
    A. If $f(a_{0})f(x_{0}) <0$, then let $[a_{1},b_{1}]$ be the next interval with $a_{1}=a_{0}$ and $b_{1}=x_{0}$.
    
    B. If $f(b_{0})f(x_{0})<0$, then let $[a_{1},b_{1}]$ be the next interval with $a_{1}=x_{0}$ and $b_{1}=b_{0}$.
1. Repeat (2) and (3) until the interval $[a_{N},b_{N}]$ reaches some predetermined length.
1. Return the value $x_{n}$, the $x$-intercept of the $N$ th subinterval.

A solution of the equation $f(x)=0$ in the interval $[a,b]$ is guaranteed by the Intermediate Value Theorem provided $f(x)$ is continuous on $[a,b]$ and $f(a)f(b)<0$. In other words, the function changes sign over the interval and therefore must equal 0 at some point in the interval $[a,b]$.

# Newton
## Formula
Let $f(x)$ be a differentiable function. If $x_{0}$ is near a solution of $f(x)=0$ then we can approximate $f(x)$ by the tangent line at $x_{0}$ and compute the $x$-intercept of the tangent line.
The equatoin of the tangent line at $x_{0}$ is $$y=f^{\prime}(x_{0})(x-x_{0}) + f(x_{0})$$
The $x$-intercept is the solution $x_{1}$ of the equation $$0=f^{\prime}(x_{1}-x_{0})+f(x_{0})$$ and we solve for $x_{1}$
$$ x_{1}=x_{0} - \frac{f(x_{0})}{f^{\prime}(x_{0})}$$
If we implement this procedure repeatedly, then we obtain a sequence given by the recurvsive formula $$x_{n+1} = x_{n} - \frac{f(x_{n})}{f^{\prime}(x_{n})}$$
which (potentially) converges to a solution of the equation $f(x)=0$.

# Definite Error
## Error Formulas
A Riemann sum is an approximation of a definite integral. A natural question arises: how good of an approximation is a Riemann sum?

**Theorem.** Let $L_{N}(f)$ denote the left Riemann sum $$L_{N}(f) = \sum_{i=1}^{N} f(x_{i-1})\Delta x$$
where $\Delta = (b-a)/N$ and $x_{i} = a+i\Delta x$. The error bound is $$E_{N}^{L} = \left| \int_{a}^{b} f(x)dx - L_{N}(f)\right| \leq \frac{\left(b-a\right)^{2}}{2N}K_{1}$$
where $\left|f^{\prime}(x)\right| \leq K_{1}$ for all $x \in \left[a,b\right]$.

**Theorem.** Let $R_{N}(f)$ denote the right Riemann sum $$R_{N}(f) = \sum_{i=1}^{N}f(x_{i})\Delta x$$
where $\Delta x = (b-a)/N$ and $x_{i} = a+i\Delta x$. The error bound is $$E_{N}^{R} = \left| \int_{a}^{b} f(x)dx - R_{N}(f)\right| \leq \frac{\left(b-a\right)^{2}}{2N}K_{1}$$
where $\left|f^{\prime}(x)\right| \leq K_{1}$ for all $x \in \left[a,b\right]$.

**Theorem.** Let $M_{N}(f)$ denote the midpoint Riemann sum $$ M_{N}(f) = \sum_{i=1}^{N}f(x_{i}^{*})\Delta x$$
where $\Delta x = (b-a)/N$ and $x_{i}^{*} = (x_{i-1}+x_{i})/2$ for $x_{i}=a+i \Delta x$. The error bound is 
$$E_{N}^{M} = \left| \int_{a}^{b} f(x)dx - M_{N}(f)\right| \leq \frac{\left(b-a\right)^{3}}{24N^{2}}K_{2}$$
where $\left|f^{\prime\prime} (x)\right| \leq K_{2} $ for all $x \in \left[a, b \right]$.

## Trapezoidal error
Error Formula
When computing integrals numerically, it is essentially to know how good our approximations are. Notice in the theorem below that the error formula is inversely proportional to $N^{2}$. This means that the error decreases much faster with larger $N$ compared to Riemann sums.

**Theorem.** Left $T_{N}(f)$ denote the trapezoid rule $$T_{N}(f) = \frac{\Delta x}{2} \sum_{i=1}^{N}\left(f(x_{i})+f(x_{i-1})\right)$$
where $\Delta x = (b-a)/N$ and $x_{i} = a+i \Delta x$. The error bound is 
$$E_{N}(f) = \left| \int_{a}^{b} f(x) dx - T_{N}(f)\right| \leq \frac{\left(b-a\right)^{3}}{12N^{2}}K_{2}$$
where $\left|f^{\prime \prime}(x)\right| \leq K_{2}$ for all $x \in \left[a, b\right]$.

## Simpson
We have seen that the error in a Riemann sum is inversely proportional to the size of the partition $N$ and the trapezoid rule is inversely proportional to $N^{2}$. The error formula in the theorem below shows that Simpson's rule is even better as the error is inversely proportional $N^{4}$.

**Theorem** Let $S_{N}(f)$ denote Simpson's rule $$S_{N}f = \frac{\Delta}{3}\sum_{i=1}^{N/2}\left(f(x_{2i-2})+4f(x_{2i-1})+f(x_{2i})\right)$$ where $N$ is an even number of subintervals of
$\left[a, b\right]$, $\Delta x = (b-a)/N$ and $x_{i}=a+i\Delta x$. The error bound is
$$E_{N}^{S}(f)=\left|\int_{a}^{b}f(x)dx - S_{N}(f)\right| \leq \frac{(b-a)^{5}}{180N^{4}}k_{4}$$
where $\left|f^{(4)}\right| \leq K_{4}$ for all $x \in \left[a, b\right]$.

