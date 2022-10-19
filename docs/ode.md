---
layout: default
title:  "ODE"
date:   2022-09-16 21:22:34 +0900
usemathjax: true
# nav_order : 1
# collection: 
# parent: 예전 자료
has_toc: true
nav_exclude: true
permalink: /ode
---
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


# Euler's Method
The simplest numerical method for approximating solutions of the differential equations is Euler's method. Consider a first order differential equation with an intitial condition:
$$y^{\prime} = f(t,y), y(t_{0}) =y_{0} $$
The idea behind Euler's method is:
1. Construct the equaion of the tangent line to the unknown function $y(t)$ at $t=t_{0}$: $$y = y(t_{0}) + f(t_{0},y_{0})(t-t_{0})$$ where $y^{\prime}(t_{0})=f(t_{0},y_{0})$ is the slope of $y(t)$ at $t=t_{0}$.

2. Use the tangent line to approximate $y(t)$ at a small time step $t_{1}=t_{0}+h$: $$y_{1} =y_{0} + f(t_{0},y_{0})(t_1{1}-t_{0})$$ where $y_{1} \approx y(t_{1})$.

3. Repeat!

The formula for Euler's method defines a recursive sequence: $$y_{n+1}=y_{n}+f(t_{n},y_{n})(t_{n+1}-t_{n}), y_{0} = y(t_{0})$$
where $y_{n} \approx y(t_{n})$ for each $n$. If we choose equally spaced $t$ values the nthe formula becomes:
$$y_{n+1} = y_{n} + f(t_{n},y_{n})h, y_{0} =y(t_{0}), t_{n} =t_{0}+nh$$
with time step $h=t_{n+1}-t_{n}$. If we implement $N$ iterations of Euler's method from $t_{0}$ to $t_{f}$ the nth time step is $$h = \frac{t_{f}-t_{0}}{N}$$
Note two very important points about Euler's method and numerical methods in general:
* A smaller time step $h$ reduces the error in the approximation.
* A smaller time step $h$ requires more computations!

# Heun's Method
Euler's method uses the degree 1 Taylor polynomial (i.e. the tangent line) to approximate $y(t)$. Heun's method is constructed from the degree 2 Taylor polynomial:
$$y(t+h) \approx y(t) + y^{\prime}(t)h+ \frac{y^{\prime \prime}}{2}h^{2}$$
We only want to use $y^{\prime}=f(t,y)$ in our approximation therefore introduce the forward difference formrula to approximate $y^{\prime \prime}$ in terms of $y^{\prime}$
$$y^{\prime \prime} = \frac{y^{\prime}(t+h)-y^{\prime}(t)}{h}$$
Put these together to approximate $y(t+h)$: 
$$y(t+h) \approx y(t) + \frac{y^{\prime}(t+h)+y^{\prime}(t)}{2}h$$
We know to how to paproximate $y^{\prime}(t)$. What about $y^{\prime}(t+h)$?
Use Euler's method $y(t+h)\approx y(t) +y^{\prime}(t)h$ to approximate;
$$y^{\prime} (t+h) = f(t+h, y(t+h)) \approx f(t+h, y(t)+y^{\prime}(t)h)$$
Heun's method is given by the 2-stage recursive formula:
$$ h = t_{n+1} -t_{n} \\ k_{1} = f(t_{n},y_{n}) \\ k_{2} = f(t_{n}+h, y_{n}+k_{1}h)\\y_{n+1} = y_{n} + \left(\frac{k_{1}+k_{2}}{2}\right)h$$

# RK4 Method
Let an IVP be specified as follows:
$$\frac{dy}{dt} = f(t,y) , y(t_{0})=y_{0}.$$
Then, the RK4 method is $$y_{n+1} = y_{n} +\frac{1}{6}(k_{1}+2k_{2}+2k_{3}+k_{4})h, \\ t_{n+1} =t_{n}+h$$
where $$k_{1} = f(t_{n},y_{n}), \\ k_{2} = f(t_{n}+\frac{h}{2}, y_{n}+h\frac{k1}{2}),\\
k3 = f(t_{n}+\frac{h}{2}, y_{n}+h\frac{k_{2}}{2}), \\ k_{4} = f(t_{n}+h, y_{n}+hk_{3}).$$
Here $y_{n+1}$ is the RK4 approximation of $y(t_{n+1})$, and the next value $(y_{n+1})$ is determined by the present value $(y_{n})$ plus the weighted average of four increments, where each increment is the product of the size of the interval, "$h$", and an estimated slope specified by function "$f$" on the right-hand side of the differential equation.

In averaging the four slopes, greater weight is given to the slopes at the midpoint. If $f$ is independent of $y$, so that the differential equation is equivalent to a simple integral ,then RK4 is Simpson's rule.

# Error
#### Order of Accuracy
Let $y_{1}$ be the approximation of $y(t_{1})$ by one step of some numerical method using step size $h=t_{1}-t_{0}$. The **(local) truncation error** (fro the given differential equation and method) is $$E(h)=\left| y(t_{1})-y_{1}\right|$$
The word *local* means we are looking at just one step of the method and the word *truncation* has to do with runcating the Taylor series.\\
Most numerical methods are based on Taylor series therefore the error may be expressed in terms of Taylors' theorem. For example, consider the Taylor series up to order $p$ evaluated at $t_{1} = t_{0}+h$:
$$y(t_{1})=y(t_{0}) + y^{\prime}(t_{0})h+ \cdots + \frac{y^{(p)}(t_{0})}{p!}h^{p} + \frac{y^{(p+1)}(c)}{(p+1)!}h^{p+1}$$
for some $ c \in \left[t_{0}, t_{1}\right]$. If a numerical method computes $y(t_{1})$ using the Taylor polynomial of degree $p$ then the local truncation error is
$$E(h)=\left|y(t_{1})-y_{1}\right| = \frac{\left|y^{(p+1)}(c)\right|}{(p+1)!}h^{p+1}$$
Threrefore we can roughly say that a numerical method is order $p$ if the local truncation error looks like $Ch^{p+1}$ for some constant $C$.
More precisely, a numerical method is **order** $p$ if the local truncation error satisfies 
$$E(h)\leq Ch^{p+1}$$
for any equation $y^{\prime}=f(t,y), y(t_{0}) = y_{0}$. The constant $C$ depends on $f$. Note that the order is a positive integer.\\
It is usually quite difficult to determine the order of a numerical method given th formula. Instead, we can determine the order experimentally. The idea is that the local truncation error should satisfy $$E(h) \approx Ch^{p+1}$$
when applied to most differential equations. Therefore we may observe the slope in the log plot:
$$\log{(E(h))} \approx (p+1)\log{h}+\log{C}$$
The procedure to experimentally determine the order $p$ of a numerical method is:
1. Apply the numerical method to the equation $y^{\prime}=y, y(0)=1$ for different steps size $h_{1}$ and $h_{2}$.
1. Compute the local truncation errors $E(h_{1})$ and $E(h_{2})$ using the exact solution $y(t)=e^{t}$.
1. Compute the slope of the log plot: $$p+1\approx \frac{\log{(E(h_{2}))} - \log{(E(h_{1}))}}{\log{h_2}-\log{h_1}}$$