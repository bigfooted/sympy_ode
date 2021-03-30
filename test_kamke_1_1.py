#
# test for kamke ODEs, for first order and first degree 
# 
#
r"""
This file contains a list of the Kamke ODEs of first order and first degree, see:
Kamke, Differentialgleichungen, loesungsmethoden und loesungen, Leipzig 1967  

The database can be used to test the solver performance of dsolve.
It can be run from the konsole using:
$ py.test --timeout=20 --durations=0 -v -s test_kamke1_1.py

in verbose mode, the test will output the solution, if found.

Functions that are for internal use:
-

"""
from sympy import (acos, asin, atan, cos, Derivative, Dummy, diff,Integral,
    E, Eq, exp, I, log, pi, Piecewise, Rational, S, sin, sinh, tan,
    sqrt, symbols, Ei, erfi, Ne)

from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
    SingleODEProblem, SingleODESolver)

from sympy.solvers.ode.subscheck import checkodesol

from sympy.testing.pytest import raises, slow
import traceback
from time import process_time
import pytest

x = Symbol('x')

u = Symbol('u')
i = Symbol('i')
y = Function('y')(x)
# arbitrary functions in the kamke ODEs
f = Function('f')
g = Function('g')
h = Function('h')

C1, C2, C3, C4, C5 = symbols('C1:6')
a0,a1,a2,a3,a4,a,b,c,A,B,C,nu,m,n = symbols('a0, a1, a2, a3, a4, a, b, c, A, B, C, nu, m, n')

kamke1_1 = [
0,
#/* 1 */
y.diff(x)-(a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**(-1/2),
#/* 2 */
y.diff(x)+a*y-c*exp(b*x),
#/* 3 */
y.diff(x)+a*y-b*sin(c*x),
#/* 4 */
y.diff(x)+2*x*y-x*exp(-x**2),
#/* 5 */
y.diff(x)+y*cos(x)-exp(2*x),
#/* 6 */
y.diff(x)+y*cos(x)-sin(2*x)/2,
#/* 7 */
y.diff(x)+y*cos(x)-exp(-sin(x)),
#/* 8 */
y.diff(x) + y*tan(x) - sin(2*x),
#/* 9 */
y.diff(x)-(sin(log(x))+cos(log(x))+a)*y,
#/* 10 */
y.diff(x) + f(x).diff(x)*y - f(x)*f(x).diff(x),
#/* 11 */
y.diff(x)  + f(x)*y - g(x),
#/* 12 */
y.diff(x) + y**2 - 1,
#/* 13 */
y.diff(x) + y**2 - a*x - b,
#/* 14 */
y.diff(x) + y**2 + a*x**m,
#/* 15 */
y.diff(x) + y**2 - 2*x**2*y + x**4 -2*x-1,
#/* 16 */
y.diff(x) + y**2 +(x*y-1)*f(x),
#/* 17 */
y.diff(x) - y**2 -3*y + 4,
#/* 18 */
y.diff(x)-y**2-x*y-x+1,
#/* 19 */
y.diff(x) - (y + x)**2,
#/* 20 */
y.diff(x)-y**2+(x**2+1)*y-2*x,
#/* 21 */
y.diff(x)-y**2+y*sin(x)-cos(x),
#/* 22 */
y.diff(x)-y**2-y*sin(2*x)-cos(2*x),
#/* 23 */
y.diff(x) + a*y**2 - b,
#/* 24 */
y.diff(x) + a*y**2 - b*x**nu,
#/* 25 */
y.diff(x)+a*y**2-b*x**(2*nu)-c*x**(nu-1),
#/* 26 */
y.diff(x)-(A*y- a)*(B*y-b),
#/* 27 */
y.diff(x) + a*y*(y-x) - 1,
#/* 28 */
y.diff(x)+x*y**2-x**3*y-2*x,
#/* 29 */
y.diff(x) - x*y**2 - 3*x*y,
#/* 30 */
y.diff(x)+x**(-a-1)*y**2-x**a,
#/* 31 */
#/* only if n # -1 */
y.diff(x) - a*x**n*(y**2+1), 
#/* 32 */
y.diff(x) + y**2*sin(x) - 2*sin(x)/cos(x)**2,
#/* 33 */
y.diff(x)-y**2*f(x).diff(x)/g(x)+g(x).diff(x)/f(x),
#/* 34 */
y.diff(x)+f(x)*y**2+g(x)*y,
#/* 35 */
y.diff(x)+f(x)*(y**2+2*a*y+b),
#/* 36 */
y.diff(x) + y**3 + a*x*y**2,
#/* 37 */
y.diff(x)-y**3-a*exp(x)*y**2,
#/* 38 */
y.diff(x) - a*y**3 - b*x**(-3/2),
#/* 39 */
y.diff(x)-a3*y**3-a2*y**2-a1*y-a0,
#/* 40 */
y.diff(x)+3*a*y**3+6*a*x*y**2,
#/* 41 */
y.diff(x)+a*x*y**3+b*y**2,
#/* 42 */
y.diff(x)-x*(x+2)*y**3-(x+3)*y**2,
#/* 43 */
y.diff(x)+(3*a*x**2+4*a**2*x+b)*y**3+3*x*y**2,
#/* 44 */
y.diff(x)+2*a*x**3*y**3+2*x*y,
#/* 45 */
y.diff(x)+2*(a**2*x**3-b**2*x)*y**3+3*b*y**2,
#/* 46 */
y.diff(x)- x**a*y**3+3*y**2-x**(-a)*y-x**(-2*a)+ a*x**(-a-1),
#/* 47 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x) - a*(x**n - x)*y**3 - y**2,*/
0,
#/* 48 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x) - (a*x**n+b*x)*y**3 - c*y**2, */
0,
#/* 49 this is actually a Weierstrass equation */
y.diff(x) - a*f(x).diff(x)*y**3 - 6*a*f(x)*y**2 - (2*a+1)*(f(x).diff(x,2)/f(x).diff(x))*y - 2*(a+1),
#/* 50 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x) = f3(x)*y**3 + f2(x)*y**2 + f1(x)*y+f0(x),*/
0,
#/* 51 */
y.diff(x) = (y-f(x))*(y-g(x))*(y-(a*f(x)+b*g(x))/(a+b))*h(x) + (y-g(x))/(f(x)-g(x))*diff(f(x),x)+(y-f(x))/(g(x)-f(x))*diff(g(x),x),
#/* 52 n is integer */
y.diff(x)-a*y**n-b*x**(n/(1-n)),
#/* 53 */
y.diff(x)-f(x)**(1-n)*g(x).diff(x)*y**n/(a*g(x)+b)**n-f(x).diff(x)*y/f(x)-f(x)*g(x).diff(g(x),x),
#/* 54 */
y.diff(x)-a**n*f(x)**(1-n)*g(x).diff(x)*y**n-f(x).diff(x)*y/f(x)-f(x)*g(x).diff(x),
#/* 55 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x) = f(x)*y**n + g(x)*y + h(x),*/
0,
#/* 56 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x)+f(x)*y**a + g(x)*y**b,*/
0,
#/* 57 */
y.diff(x)-sqrt(abs(y)),
#/* 58 */
y.diff(x)-a*sqrt(y)-b*x,
#/* 59 */
y.diff(x)-a*sqrt(y**2+1)-b,
#/* 60 */
y.diff(x)-sqrt(y**2-1)/sqrt(x**2-1),
#/* 61 */
y.diff(x)-sqrt(x**2-1)/sqrt(y**2-1),
#/* 62 */
y.diff(x)=(y-x**2*sqrt(x**2-y**2))/(x*y*sqrt(x**2-y**2)+x),
#/* 63 NOTE: no abs-sign here! */
y.diff(x)-(1+ y**2)/((y+sqrt(1+y))*sqrt(1+x)**3),
#/* 64 */
y.diff(x)-sqrt((a*y**2+b*y+c)/(a*x**2+b*x+c)),
#/* 65 */
y.diff(x)-sqrt(y**3+1)/sqrt(x**3+1),
#/* 66 NOTE: no abs sign here!*/
#/*y.diff(x)-sqrt(abs(y*(1-y)*(1-a*y)))/sqrt(abs(x*(1-x)*(1-a*x))),*/
y.diff(x)-(sqrt(y*(1-y)*(1-a*y)))/(sqrt(x*(1-x)*(1-a*x))),
#/* 67 */
y.diff(x)-sqrt(1-y**4)/sqrt(1-x**4),
#/* 68 */
y.diff(x)-sqrt((a*y**4+b*y**2+1)/(a*x**4+b*x**2+1)),
#/* 69 */ /* nijso bug: missing a0,b0 */
y.diff(x)=sqrt((a0 + a1*x**1 + a2*x**2 + a3*x**3 + a4*x**4)*(b0 +b1*y**1+b2*y**2+b3*y**3+b4*y**4)),
#/* 70 */ /* nijso bug: missing a0,b0 */
y.diff(x)=sqrt((a0 + a1*x**1 + a2*x**2 + a3*x**3 + a4*x**4)/(b0 + b1*y**1+b2*y**2+b3*y**3+b4*y**4)),
#/* 71 */ /* *nijso BUG: missing b0,a0 */
y.diff(x)=sqrt((b0 + b1*y**1 + b2*y**2 + b3*y**3 + b4*y**4)/(a0 + a1*x**1+b2*x**2+b3*x**3+b4*x**4)),
#/* 72  y'=R1(x,sqrt(X))*R2(y,sqrt(Y)) with R1,R2 rational functions, here an example */
y.diff(x)=(y/sqrt(b1*y**1 + b2*y**2 + b3*y**3 + b4*y**4))*(x/sqrt(a1*x**1+b2*x**2+b3*x**3+b4*x**4)),
#/* 73 */ /* nijso bug added a0,b0, removed b4,a4*/
y.diff(x)=(b0 + b1*y**1+b2*y**2+b3*y**3)**(2/3)/(a0 + a1*x**1 + a2*x**2 + a3*x**3)**(2/3),
#/* 74 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x)=f(x)*(y-g(x))*sqrt((y-a)*(y-b)),*/
0,
#/* 75 */
y.diff(x)-exp(x-y)+exp(x),
#/* 76 */
y.diff(x)-a*cos(y)+b,
#/* 77 */
y.diff(x)=cos(a*y+b*x),
#/* 78 */
y.diff(x)+a*sin(a1*y+b1*x)+b,
#/* 79 - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*y.diff(x)+f(x)*cos(a*y)+g(x)*sin(a*y)+h(x),*/
0,
#/* 80 */
y.diff(x)+f(x)*sin(y)+(1-f(x).diff(x))*cos(y)-f(x).diff(x)-1,
#/* 81 */
y.diff(x)+2*tan(y)*tan(x),
#/* 82 - Too general - E S Cheb-Terrab and T Kolokolnikov (I am also not sure if atan(x) is meant here) */
/*y.diff(x)=a*(1+tan(y)**2) + tan(y)*atan(x),*/
0,
#/* 83 */
y.diff(x)-tan(x*y),
#/* 84 */
y.diff(x)-f(a*x+b*y),
#/* 85 */
y.diff(x)=x**(a-1)*y**(1-b)*f(x**a/a + y**b/b),
#/* 86 */
y.diff(x)=(y-x*f(x**2+a*y**2))/(x+a*y*f(x**2+a*y**2)),
#/* 87 */
y.diff(x)=(y/x)*(a*f(x**c*y)+c*x**a*y**b)/(b*f(x**c*y)-x**a*y**b),
#/* 88 */
2*y.diff(x)-3*y**2-4*a*y-b-c*exp(-2*a*x),
#/* 89 */
 x*y.diff(x)-sqrt(a**2-x**2),
#/* 90 */
x*y.diff(x)+y-x*sin(x),
#/* 91 */
x*y.diff(x)-y-x/log(x),
#/* 92 */
x*y.diff(x)-y-x**2*sin(x),
#/* 93 */
x*y.diff(x)-y-x*cos(log(log(x)))/log(x),
#/* 94 */
x*y.diff(x)+a*y+b*x**n,
#/* 95 */
x*y.diff(x)+y**2+x**2,
#/* 96 */
x*y.diff(x)-y**2+1,
#/* 97 */
x*y.diff(x)+a*y**2-y+b*x**2,
#/* 98 */ 
x*y.diff(x)+a*y**2-b*y+c*x**(2*b),
#/* 99 */
x*y.diff(x)+a*y**2-b*y-c*x**Beta,
#/* 100 */
x*y.diff(x)+x*y**2+a,
#/* 101 */
x*y(x).diff(x)+x*y**2-y,
#/* 102 */
x*y(x).diff(x)+x*y**2-y-a*x**3,
#/* 103  (nijso: minus sign error in database)*/
x*y(x).diff(x)-x*y**2-(2*x**2+1)*y-x**3,
#/* 104 */
x*y(x).diff(x)+a*x*y**2+2*y+b*x,
#/* 105 */
x*y(x).diff(x)+a*x*y**2+b*y+c*x+d,
#/* 106 */
x*y(x).diff(x)+x**a*y**2+(a-b)*y/2+x**b,
#/* 107 */
x*y(x).diff(x)+a*x**Alpha*y**2+b*y-c*x**Beta,
#/* 108 */
x*y(x).diff(x)-y**2*log(x)+y,
#/* 110 */
x*y(x).diff(x)-y*(2*y*log(x)-1),
#/* 110 */
x*y(x).diff(x)+ f(x)*(y**2-x**2)-y,
#/* 111 */
x*y(x).diff(x) + y**3 + 3*x*y**2,
#/* 112 */
x*y(x).diff(x)-sqrt(y**2+x**2)-y,
#/* 113 */
x*y(x).diff(x)+a*sqrt(y**2+x**2)-y,
#/* 114 */
x*y(x).diff(x)-x*sqrt(y**2+x**2)-y,
#/* 115 */
x*y(x).diff(x)-x*(y-x)*sqrt(y**2+x**2)-y,
#/* 116 */
x*y(x).diff(x)-x*sqrt((y**2-x**2)*(y**2-4*x**2))-y, 
#/* 117 */
x*y(x).diff(x)-x*exp(y/x)-y-x,
#/* 118 */
x*y(x).diff(x)-y*log(y),
#/* 119 */
x*y(x).diff(x)-y*(log(x*y)-1),
#/* 120  */
x*y(x).diff(x)-y*(x*log(x**2/y)+2),
#/* 121 */ 
x*y(x).diff(x)+sin(y-x),
#/* 122 */
x*y(x).diff(x)+(sin(y)-3*x**2*cos(y))*cos(y),
#/* 123 */
x*y(x).diff(x)-x*sin(y/x)-y,
#/* 124 */
x*y(x).diff(x)+x*cos(y/x)-y+x,
#/* 125 */
x*y(x).diff(x)+x*tan(y/x)-y,
#/* 126 */
x*y(x).diff(x)-y*f(x*y),
#/* 127  */
x*y(x).diff(x)-y*f(x**a*y**b),
#/* 128  */
x*y(x).diff(x)+a*y-f(x)*g(x**a*y),
#/* 129 */
(x+1)*y(x).diff(x)+y*(y-x),
#/* 130 */
2*x*y(x).diff(x)-y-2*x**3,
#/*  131  */ 
(2*x+1)*y(x).diff(x)-4*%e**-y+2,
#/*  132  */ 
3*x*y(x).diff(x)-3*x*log(x)*y**4-y,
#/*  133  */ 
x**2*y(x).diff(x)+y-x,
#/*  134  */ 
x**2*y(x).diff(x)-y+x**2*%e**(x-1/x),
#/*  135  */ 
x**2*y(x).diff(x)-(x-1)*y,
#/*  136  */ 
x**2*y(x).diff(x)+y**2+x*y+x**2,
#/*  137  */ 
x**2*y(x).diff(x)-y**2-x*y,
#/*  138  */ 
x**2*y(x).diff(x)-y**2-x*y-x**2,
#/*  139  */ 
x**2*(y(x).diff(x)+y**2)+a*x**k-(b-1)*b,
#/*  140  */ 
x**2*(y(x).diff(x)+y**2)+4*x*y+2,
#/*  141  */ 
x**2*(y(x).diff(x)+y**2)+a*x*y+b,
#/*  142  */ 
x**2*(y(x).diff(x)-y**2)-a*x**2*y+a*x+2,
#/*  143  */ 
x**2*(y(x).diff(x)+a*y**2)-b,
#/*  144  */ 
x**2*(y(x).diff(x)+a*y**2)+b*x**Alpha+c,
#/*  145  */ 
x**2*y(x).diff(x)+a*y**3-a*x**2*y**2,
#/*  146  */ 
x**2*y(x).diff(x)+x*y**3+a*y**2,
#/*  147  */ 
x**2*y(x).diff(x)+a*x**2*y**3+b*y**2,
#/*  148  */ 
(x**2+1)*y(x).diff(x)+x*y-1,
#/*  149  */ 
(x**2+1)*y(x).diff(x)+x*y-x*(x**2+1),
#/*  150  */ 
(x**2+1)*y(x).diff(x)+2*x*y-2*x**2,
#/*  151  */ 
(x**2+1)*y(x).diff(x)+(2*x*y-1)*(y**2+1),
#/*  152  */ 
(x**2+1)*y(x).diff(x)+x*cos(y)*sin(y)-x*(x**2+1)*cos(y)**2,
#/*  153  */ 
(x**2-1)*y(x).diff(x)-x*y+a,
#/*  154  */ 
(x**2-1)*y(x).diff(x)+2*x*y-cos(x),
#/*  155  */ 
(x**2-1)*y(x).diff(x)+y**2-2*x*y+1,
#/*  156  */ 
(x**2-1)*y(x).diff(x)-y*(y-x),
#/*  157  */ 
(x**2-1)*y(x).diff(x)+a*(y**2-2*x*y+1),
#/*  158  */ 
(x**2-1)*y(x).diff(x)+a*x*y**2+x*y,
#/*  159  */ 
(x**2-1)*y(x).diff(x)-2*x*y*log(y),
#/*  160  */ 
(x**2-4)*y(x).diff(x)+(x+2)*y**2-4*y,
#/*  161  */ 
(x**2-5*x+6)*y(x).diff(x)+3*x*y-8*y+x**2,
#/*  162  */ 
(x-a)*(x-b)*y(x).diff(x)+y**2+k*(y+x-a)*(y+x-b),
#/*  163  */
2*x**2*y(x).diff(x)-2*y**2-x*y+2*a**2*x,
#/*  164  */ 
2*x**2*y(x).diff(x)-2*y**2-3*x*y+2*a**2*x,
#/*  165  */ 
x*(2*x-1)*y(x).diff(x)+y**2+(-4*x-1)*y+4*x,
#/*  166  */ 
2*(x-1)*x*y(x).diff(x)+(x-1)*y**2-x,
#/*  167  */ 
3*x**2*y(x).diff(x)-7*y**2-3*x*y-x**2,
#/*  168  */ 
3*(x**2-4)*y(x).diff(x)+y**2-x*y-3,
#/*  169  */ 
(a*x+b)**2*y(x).diff(x)+(a*x+b)*y**3+c*y**2,
#/*  170  */ 
x**3*y(x).diff(x)-y**2-x**4,
#/*  171  */ 
x**3*y(x).diff(x)-y**2-x**2*y,
#/*  172  */ 
x**3*y(x).diff(x)-x**4*y**2+x**2*y+20,
#/*  173  */ 
x**3*y(x).diff(x)-x**6*y**2+(3-2*x)*x**2*y+3,
#/*  174  */ 
x*(x**2+1)*y(x).diff(x)+x**2*y,
#/*  175  */ 
x*(x**2-1)*y(x).diff(x)-(2*x**2-1)*y+a*x**3,
#/*  176  */ 
x*(x**2-1)*y(x).diff(x)+(x**2-1)*y**2-x**2,
#/*  177  */ 
(x-1)*x**2*y(x).diff(x)-y**2-(x-2)*x*y,
#/*  178  */ 
2*x*(x**2-1)*y(x).diff(x)+2*(x**2-1)*y**2+(5-3*x**2)*y+x**2-3,
#/*  179  */ 
3*x*(x**2-1)*y(x).diff(x)+x*y**2+(-x**2-1)*y-3*x,
#/*  180  */ 
(a*x**2+b*x+c)*(x*y(x).diff(x)-y)-y**2+x**2,
#/*  181  */ 
x**4*(y(x).diff(x)+y**2)+a,
#/*  182  */ 
x*(x**3-1)*y(x).diff(x)-2*x*y**2+y+x**2,
#/*  183  */ 
(2*x**4-x)*y(x).diff(x)-2*(x**3-1)*y,
#/* 184 */
(a*x**2+b*x+c)**2*(y(x).diff(x)+y**2)+A,
#/*  185  */ 
x**7*y(x).diff(x)+2*(x**2+1)*y**3+5*x**3*y**2 , 
#/*  186  */ 
x**n*y(x).diff(x)+y**2+(1-n)*x**(n-1)*y+x**(2*n-2),
#/*  187  */ 
x**n*y(x).diff(x)-a*y**2-b*x**(2*n-2),
#/* 188  Abel eqn
#  Some choices that are integrable include
#    (3, b:1, a:n+b);   => K = -27/4
#    (7, b:2, a:n+b);   => K = -343/36
#*/
x**(2*n+1)*y(x).diff(x)-a*y**3-b*x**(3*n),
#/*  189  */ 
x**(n+m*(n-1))*y(x).diff(x)-a*y**n-b*x**((m+1)*n) , 
#/*  190  */ 
sqrt(x**2-1)*y(x).diff(x)-sqrt(y**2-1),
#/*  191  */ 
sqrt(1-x**2)*y(x).diff(x)-y*sqrt(y**2-1),
#/*  192  */ 
sqrt(x**2+a**2)*y(x).diff(x)+y-sqrt(x**2+a**2)+x,
#/*  193  */ 
x*log(x)*y(x).diff(x)+y-a*x*(log(x)+1),
#/*  194  */ 
x*log(x)*y(x).diff(x)-log(x)*y**2+(-2*log(x)**2-1)*y-log(x)**3,
#/*  195  */ 
sin(x)*y(x).diff(x)-sin(x)**2*y**2+(cos(x)-3*sin(x))*y+4,
#/*  196  */ 
cos(x)*y(x).diff(x)+y+cos(x)*(sin(x)+1),
#/*  197  */ 
cos(x)*y(x).diff(x)-y**4-sin(x)*y,
#/*  198  */
cos(x)*sin(x)*y(x).diff(x)-y-sin(x)**3,
#/*  199  - also Murphy 1.129 */ 
sin(2*y)+sin(2*x)*y(x).diff(x),
#/*  200  */ 
(a*sin(x)**2+b)*y(x).diff(x)+a*sin(2*x)*y+A*x*(a*sin(x)**2+c),
#/*  201  */ 
2*f(x)*y(x).diff(x)+2*f(x)*y**2-f(x).diff(x)*y-2*f(x)**2,
#/*  202  - Too general - E S Cheb-Terrab and T Kolokolnikov */
#/*f(x)*y(x).diff(x)+g(x)*tan(y)+h(x),*/
0,
#/*  203  */ 
y*y(x).diff(x)+y+x**3,
#/*  204  */ 
y*y(x).diff(x)+a*y+x,
#/*  205  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*y*y(x).diff(x)+a*y+b*x**n+(a**2-1)*x/4 ,  */
0,
#/*  206  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*y*y(x).diff(x)+a*y+b*%e**x-2*a ,  */
0,
#/*  207  */ 
y*y(x).diff(x)+y**2+4*x*(x+1),
#/*  208  */ 
y*y(x).diff(x)+a*y**2-b*cos(x+c),
#/*  209  */ 
y*y(x).diff(x)-sqrt(a*y**2+b),
#/*  210  */ 
y*y(x).diff(x)+x*y**2-4*x,
#/*  211  */ 
y*y(x).diff(x)-x*%e**(x/y),
#/*  212  */ 
g(x)*f(y**2+x**2)+y*y(x).diff(x)+x ,  
#/*  213  */ 
(y+1)*y(x).diff(x)=y+x,
#/*  214  */ 
(y+x-1)*y(x).diff(x)-y+2*x+3,
#/*  215  */ 
(y+2*x-2)*y(x).diff(x)-y+x+1,
#/*  216  */ 
(y-2*x+1)*y(x).diff(x)+y+x,
#/*  217  */ 
(y-x**2)*y(x).diff(x)=x,
#/*  218  */ 
(y-x**2)*y(x).diff(x)+4*x*y,
#/*  219  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*(y+g(x))*y(x).diff(x)-f2(x)*y**2-f1(x)*y-f0(x) ,  */
0,
#/*  220  */ 
2*y*y(x).diff(x)-x*y**2-x**3,
#/*  221  */ 
(2*y+x+1)*y(x).diff(x)-(2*y+x-1),
#/*  222  */ 
(2*y+x+7)*y(x).diff(x)-y+2*x+4,
#/*  223  */ 
(2*y-x)*y(x).diff(x)-y-2*x,
#/*  224  */ 
(2*y-6*x)*y(x).diff(x)-y+3*x+2,
#/*  225  */ 
(4*y+2*x+3)*y(x).diff(x)-2*y-x-1,
#/*  226  */ 
(4*y-2*x-3)*y(x).diff(x)+2*y-x-1,
#/*  227  */ 
(4*y-3*x-5)*y(x).diff(x)-3*y+7*x+2,
#/*  228  */ 
(4*y+11*x-11)*y(x).diff(x)-25*y-8*x+62,
#/*  229  */ 
(12*y-5*x-8)*y(x).diff(x)-5*y+2*x+3,
#/*  230  */ 
a*y*y(x).diff(x)+b*y**2+f(x),
#/*  231  */ 
Gamma+(a*y+b*x+c)*y(x).diff(x)+Alpha*y+Beta*x ,  
#/*  232  */ 
x*y*y(x).diff(x)+y**2+x**2,
#/*  233  */ 
x*y*y(x).diff(x)-y**2+a*x**3*cos(x) ,
#/*  234  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*x*y*y(x).diff(x)-y**2+x*y+x**3-2*x**2 ,  */
0,
#/*  235  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*(x*y+a)*y(x).diff(x)+b*y ,  */
0,
#/*  236  */ 
x*(y+4)*y(x).diff(x)-y**2-2*y-2*x ,
#/*  237  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*x*(y+a)*y(x).diff(x)+b*y+c*x ,  */
0,
#/*  238  */ 
(x*(y+x)+a)*y(x).diff(x)-y*(y+x)-b ,  
#/*  239  */ 
(x*y-x**2)*y(x).diff(x)+y**2-3*x*y-2*x**2,
#/*  240  */ 
2*x*y*y(x).diff(x)-y**2+a*x,
#/*  241  */ 
2*x*y*y(x).diff(x)-y**2+a*x**2,
#/*  242  */ 
2*x*y*y(x).diff(x)+2*y**2+1,
#/*  243  */ 
x*(2*y+x-1)*y(x).diff(x)-y*(y+2*x+1),
#/*  244  */ 
x*(2*y-x-1)*y(x).diff(x)+(-y+2*x-1)*y,
#/*  245  */ 
(2*x*y+4*x**3)*y(x).diff(x)+y**2+112*x**2*y,
#/*  246  */ 
x*(3*y+2*x)*y(x).diff(x)+3*(y+x)**2,
#/*  247  */ 
(3*x+2)*(y-2*x-1)*y(x).diff(x)-y**2+x*y-7*x**2-9*x-3,
#/*  248  */ 
 (6*x*y+x**2+3)*y(x).diff(x)+3*y**2+2*x*y+2*x ,  
#/*  249  */ 
 (a*x*y+b*x**n)*y(x).diff(x)+Alpha*y**3+Beta*y**2 ,  
#/*  250  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
 gamma+(B*x*y+b*y+A*x**2+a*x+c)*y(x).diff(x)+A*x*y+beta*y-B*g(x)**2+alpha*x ,  
#/*  251  */ 
(x**2*y-1)*y(x).diff(x)+x*y**2-1,
#/*  252  */ 
(x**2*y-1)*y(x).diff(x)-x*y**2+1,
#/*  253  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*(x**2*y-1)*y(x).diff(x)+8*(x*y**2-1) ,  */
0,
#/*  254  */ 
x*(x*y-2)*y(x).diff(x)+x**2*y**3+x*y**2-2*y,
#/*  255  */ 
x*(x*y-3)*y(x).diff(x)+x*y**2-y,
#/*  256  */ 
x**2*(y-1)*y(x).diff(x)+(x-1)*y,
#/*  257  */ 
x*(x*y+x**4-1)*y(x).diff(x)-y*(x*y-x**4-1),
#/*  258  */ 
2*x**2*y*y(x).diff(x)+y**2-2*x**3-x**2,
#/*  259  */ 
2*x**2*y*y(x).diff(x)-y**2-x**2*%e**(x-1/x),
#/*  260  */ 
(2*x**2*y+x)*y(x).diff(x)-x**2*y**3+2*x*y**2+y,
#/*  261  */ 
(2*x**2*y-x)*y(x).diff(x)-2*x*y**2-y,
#/*  262  */ 
(2*x**2*y-x**3)*y(x).diff(x)+y**3-4*x*y**2+2*x**3,
#/*  263  */ /* nijso fixed ode */
2*x**3*y*y(x).diff(x)+3*x**2*y**2+7,
#/*  264  */ 
2*x*(x**3*y+1)*y(x).diff(x)+y*(3*x**3*y-1),
#/*  265  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*(x**(n*(n+1))*y-1)*y(x).diff(x)+2*(n+1)**2*x**(n-1)*(x**n**2*y**2-1) ,  */
0,
#/*  266  */ 
sqrt(x**2+1)*(y-x)*y(x).diff(x)-a*(y**2+1)**(3/2) ,
#/*  267  */ 
sin(x)**2*y*y(x).diff(x)+cos(x)*sin(x)*y**2-1,
#/*  268  */ 
f(x)*y*y(x).diff(x)+g(x)*y**2+h(x),
#/*  269  - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
#/*(bessel_i(1,x)*%e**-x*y+bessel_i(0,x)*%e**-x)*y(x).diff(x)-f3(x)*y**3-f2(x)*y**2-f1(x)*y-f0(x) ,  */
0,
#/*  270  */ 
(y**2-x)*y(x).diff(x)-y+x**2,
#/*  271  */ 
(y**2+x**2)*y(x).diff(x)+2*x*(y+2*x),
#/*  272  */ 
(y**2+x**2)*y(x).diff(x)-y**2,
#/*  273  */ 
(y**2+x**2+a)*y(x).diff(x)+2*x*y,
#/*  274  */ 
(y**2+x**2+a)*y(x).diff(x)+2*x*y+x**2+b,
#/*  275  */ 
(y**2+x**2+x)*y(x).diff(x)-y,
#/*  276  */ 
(y**2-x**2)*y(x).diff(x)+2*x*y,
#/*  277  */ 
(y**2+x**4)*y(x).diff(x)-4*x**3*y,
#/*  278  */ 
(y**2+4*sin(x))*y(x).diff(x)-cos(x),
#/*  279  */ 
(y**2+2*y+x)*y(x).diff(x)+y**2*(y+x)**2+y*(y+1) ,
#/*  280  */ 
(y+x)**2*y(x).diff(x)-a**2,
#/*  281  */ 
(y**2+2*x*y-x**2)*y(x).diff(x)-y**2+2*x*y+x**2,
#/*  282  */ 
(y+3*x-1)**2*y(x).diff(x)-(2*y-1)*(4*y+6*x-3),
#/*  283  */ 
3*(y**2-x**2)*y(x).diff(x)+2*y**3-6*x*(x+1)*y-3*%e**x,
#/*  284  */ 
(4*y**2+x**2)*y(x).diff(x)-x*y,
#/*  285  */ 
(4*y**2+2*x*y+3*x**2)*y(x).diff(x)+y**2+6*x*y+2*x**2,
#/*  286  */ 
(2*y-3*x+1)**2*y(x).diff(x)-(3*y-2*x-4)**2,
#/*  287  */ 
(2*y-4*x+1)**2*y(x).diff(x)-(y-2*x)**2,
#/*  288  */ 
(6*y**2-3*x**2*y+1)*y(x).diff(x)-3*x*y**2+x,
#/*  289  */ 
(6*y-x)**2*y(x).diff(x)-6*y**2+2*x*y+a,
#/*  290  */ 
(a*y**2+2*b*x*y+c*x**2)*y(x).diff(x)+b*y**2+2*c*x*y+d*x**2,
#/*  291  */ 
(b*(Beta*y+Alpha*x)**2-Beta*(b*y+a*x))*y(x).diff(x)+a*(Beta*y+Alpha*x)**2-Alpha*(b*y+a*x) ,
#/*  292  */ 
(Gamma+Alpha*y+Beta*x)**2+(a*y+b*x+c)**2*y(x).diff(x) ,  
#/*  293  */ 
x*(y**2-3*x)*y(x).diff(x)+2*y**3-5*x*y,
#/*  294  */ 
x*(y**2+x**2-a)*y(x).diff(x)-y*(y**2+x**2+a),
#/*  295  */ 
x*(y**2+x*y-x**2)*y(x).diff(x)-y**3+x*y**2+x**2*y,
#/*  296  */ 
x*(y**2+x**2*y+x**2)*y(x).diff(x)-2*y**3-2*x**2*y**2+x**4 ,  
#/*  297  */ 
2*x*(y**2+5*x**2)*y(x).diff(x)+y**3-x**2*y,
#/*  298  */ 
3*x*y**2*y(x).diff(x)+y**3-2*x,
#/*  299  */ 
(3*x*y**2-x**2)*y(x).diff(x)+y**3-2*x*y,
#/*  300  */ 
6*x*y**2*y(x).diff(x)+2*y**3+x,
#/*  301  */ 
(6*x*y**2+x**2)*y(x).diff(x)-y*(3*y**2-x),
#/*  302  */ 
(x**2*y**2+x)*y(x).diff(x)+y,
#/*  303  */ 
x*(x*y-1)**2*y(x).diff(x)+y*(x**2*y**2+1),
#/*  304  */ 
(10*x**3*y**2+x**2*y+2*x)*y(x).diff(x)+5*x**2*y**3+x*y**2,
#/*  305  */ 
(y**3-3*x)*y(x).diff(x)-3*y+x**2,
##/*  306  */ 
(y**3-x**3)*y(x).diff(x)-x**2*y,
#/*  307  */ 
y*(y**2+x**2+a)*y(x).diff(x)+x*(y**2+x**2-a),
#/*  308  */ 
2*y**3*y(x).diff(x)+x*y**2,
#/*  309  */ 
(2*y**3+y)*y(x).diff(x)-2*x**3-x,
#/*  310  */ 
(2*y**3+5*x**2*y)*y(x).diff(x)+5*x*y**2+x**3,
#/*  311  */ 
(20*y**3-3*x*y**2+6*x**2*y+3*x**3)*y(x).diff(x)-y**3+6*x*y**2+9*x**2*y+4*x**3,
#/*  312  */ 
(y**2/b+x**2/a)*(y*y(x).diff(x)+x)+(a-b)*(y*y(x).diff(x)-x)/(b+a) , 
#/*  313  */ 
 (2*a*y**3+3*a*x*y**2-b*x**3+c*x**2)*y(x).diff(x)-a*y**3+c*y**2+3*b*x**2*y+2*b*x**3 , 
#/*  314  */ 
x*y**3*y(x).diff(x)+y**4-x*sin(x),
#/*  315  */ 
(2*x*y**3-x**4)*y(x).diff(x)-y**4+2*x**3*y,
#/*  316  */ /* nijso: -4 forgotten in database!*/ 
(2*x*y**3+y)*y(x).diff(x)+2*y**2-4,
#/*  317  */ 
 (2*x*y**3+x*y+x**2)*y(x).diff(x)+y**2-x*y ,  
#/*  318  */ 
(3*x*y**3-4*x*y+y)*y(x).diff(x)+y**2*(y**2-2),
#/*  319  */ 
(7*x*y**3+y-5*x)*y(x).diff(x)+y**4-5*y,
#/*  320  */ 
(x**2*y**3+x*y)*y(x).diff(x)-1,
#/*  321  */ 
 (2*x**2*y**3+x**2*y**2-2*x)*y(x).diff(x)-2*y-1 , 
#/*  322  */ 
(10*x**2*y**3-3*y**2-2)*y(x).diff(x)+5*x*y**4+x,
#/*  323  */ 
 x*(a*x*y**3+c)*y(x).diff(x)+y*(b*x**3*y+c) , 
#/*  324  */ 
 (2*x**3*y**3-x)*y(x).diff(x)+2*x**3*y**3-y , 
#/*  325  */ 
y*(y**3-2*x**3)*y(x).diff(x)+x*(2*y**3-x**3),
#/*  326  */ 
y*((a*y+b*x)**3+b*x**3)*y(x).diff(x)+x*((a*y+b*x)**3+a*y**3),
#/*  327  */ 
 (x*y**4+2*x**2*y**3+2*y+x)*y(x).diff(x)+y**5+y , 
#/*  328  */ 
a*x**2*y**n*y(x).diff(x)-2*x*y(x).diff(x)+y,
#/*  329  */ 
x**n*y**m*(a*x*y(x).diff(x)+b*y)+Alpha*x*y(x).diff(x)+Beta*y,
#/*  330  */ 
 y(x).diff(x)*(f(y+x)+1)+f(y+x) , 
#/*  331 - Too general - E S Cheb-Terrab and T Kolokolnikov */ 
0 ,  
/*  332  */ 
x*(sqrt(x*y)-1)*y(x).diff(x)-y*(sqrt(x*y)+1),
#/*  333  */ 
 (2*x**(5/2)*y**(3/2)+x**2*y-x)*y(x).diff(x)-x**(3/2)*y**(5/2)+x*y**2-y ,  
#/*  334  */ 
 (sqrt(y+x)+1)*y(x).diff(x)+1 , 
#/*  335  */ 
sqrt(y**2-1)*y(x).diff(x)-sqrt(x**2-1),
#/*  336  */ 
(sqrt(y**2+1)+a*x)*y(x).diff(x)+a*y+sqrt(x**2+1),
#/*  337  */ 
 (sqrt(y**2+x**2)+x)*y(x).diff(x)-y ,  
#/*  338  */ 
 (y*sqrt(y**2+x**2)+sin(Alpha)*(y**2-x**2)-2*cos(Alpha)*x*y)*y(x).diff(x)+x*sqrt(y**2+x**2)+cos(Alpha)*(y**2-x**2)+2*sin(Alpha)*x*y , 
#/*  339  */ 
 (x*sqrt(y**2+x**2+1)-y*(y**2+x**2))*y(x).diff(x)-y*sqrt(y**2+x**2+1)-x*(y**2+x**2) , 
#/*  340  */ 
(e1*(x+a)/(y**2+(x+a)**2)**(3/2)+e2*(x-a)/(y**2+(x-a)**2)**(3/2))*y(x).diff(x)-y*(e1/(y**2+(x+a)**2)**(3/2)+e2/(y**2+(x-a)**2)**(3/2)),
#/*  341  */ 
(x*%e**y+%e**x)*y(x).diff(x)+%e**y+%e**x*y,
#/*  342  */ 
x*(3*%e**(x*y)+2*%e**-(x*y))*(x*y(x).diff(x)+y)+1,
#/*  343  */ 
(log(y)+x)*y(x).diff(x)-1,
#/*  344  */ 
(log(y)+2*x-1)*y(x).diff(x)-2*y,
#/*  345  */ 
x*(2*x**2*y*log(y)+1)*y(x).diff(x)-2*y,
#/*  346  */ 
 x*y(x).diff(x)*(y*log(x*y)+y-a*x)-y*(a*x*log(x*y)-y+a*x) , 
#/*  347  */ 
(sin(x)+1)*sin(y)*y(x).diff(x)+cos(x)*(cos(y)-1),
#/*  348  */ 
(x*cos(y)+sin(x))*y(x).diff(x)+sin(y)+cos(x)*y,
#/*  349  */ 
2*x*sin(y/x)+x*y(x).diff(x)*cot(y/x)-y*cot(y/x),
#/*  350  */ 
cos(y)*y(x).diff(x)-cos(x)*sin(y)**2-sin(y),
#/*  351  */ 
 cos(y)*y(x).diff(x)-sin(y)**3+x*cos(y)**2*sin(y) , 
#/*  352  */ 
cos(y)*(cos(y)-sin(Alpha)*sin(x))*y(x).diff(x)+cos(x)*(cos(x)-sin(Alpha)*sin(y)),
#/*  353  */ 
x*cos(y)*y(x).diff(x)+sin(y),
#/*  354  */ 
(x*sin(y)-1)*y(x).diff(x)+cos(y),
#/*  355  */ 
(x*cos(y)+cos(x))*y(x).diff(x)+sin(y)-sin(x)*y,
#/*  356  */ 
(x**2*cos(y)+2*sin(x)*y)*y(x).diff(x)+2*x*sin(y)+cos(x)*y**2,
#/*  357  */ 
x*log(x)*sin(y)*y(x).diff(x)+cos(y)*(1-x*cos(y)),
#/*  358  */ 
cos(x)*sin(y)*y(x).diff(x)+sin(x)*cos(y),
#/*  359  */ /* nijso fixed wrong ode*/
3*sin(x)*sin(y)*y(x).diff(x)+5*cos(x)*cos(y)**3,
#/*  360  */ 
y(x).diff(x)*cos(a*y)-b*(1-c*cos(a*y))*sqrt(cos(a*y)**2+c*cos(a*y)-1),
#/*  361  */ 
y(x).diff(x)*(cos(y+x)+x*sin(x*y)-sin(y))+cos(y+x)+y*sin(x*y)+cos(x),
#/*  362  */ 
y(x).diff(x)*(x**2*y*sin(x*y)-4*x)+x*y**2*sin(x*y)-y,
#/*  363  */ 
(x*y(x).diff(x)-y)*cos(y/x)**2+x,
#/*  364  */ 
x*y(x).diff(x)*(y*sin(y/x)-x*cos(y/x))-y*(y*sin(y/x)+x*cos(y/x)),
#/*  365  */ 
y(x).diff(x)*(y*f(y**2+x**2)-x)+x*f(y**2+x**2)+y , 
#/*  366  */ 
(a*y*y(x).diff(x)+x)*f(a*y**2+x**2)-x*y(x).diff(x)-y,
#/*  367  */ 
(b*x*y(x).diff(x)-a)*f(x**c*y)-x**a*y**b*(x*y(x).diff(x)+c*y) , 

]


# add the solution here

solution_kamke1=[
0,
#kamke1.1
Eq(y, C1 + Integral(1/sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4), x)),
#kamke1.2
Eq(y, (C1 + c*Piecewise((exp(a*x)*exp(b*x)/(a + b), Ne(a, -b)), (x, True)))*exp(-a*x)),
#kamke1.3
Eq(y, (C1 + b*Piecewise((0, Eq(a, 0) & Eq(c, 0)), (x*exp(-I*c*x)*sin(c*x)/2 - I*x*exp(-I*c*x)*cos(c*x)/2 - exp(-I*c*x)*cos(c*x)/(2*c), Eq(a, -I*c)), (x*exp(I*c*x)*sin(c*x)/2 + I*x*exp(I*c*x)*cos(c*x)/2 - exp(I*c*x)*cos(c*x)/(2*c), Eq(a, I*c)), (a*exp(a*x)*sin(c*x)/(a**2 + c**2) - c*exp(a*x)*cos(c*x)/(a**2 + c**2), True)))*exp(-a*x)),
#kamke1.4
Eq(y, (C1 + x**2/2)*exp(-x**2)),
#kamke1.5
Eq(Integral((y*cos(x) - exp(2*x))*exp(sin(x)), x), C1),
#kamke1.6
Eq(y, C1*exp(-sin(x)) + sin(x) - 1),
#kamke1.7
Eq(y, (C1 + x)*exp(-sin(x))),
#kamke1.8
Eq(y, (C1 - 2*cos(x))*cos(x)),
#kamke1.9
Eq(y, C1*exp(x*(a + sin(log(x))))),
#kamke1.10
Eq(y, C1*exp(-f(x)) + f(x) - 1)
]



# list of kamke ODEs that depend on arbitrary functions. We need this for checkodesol
kamkefunctions = [0,
None,None,None,None,None,None,None,None,None,[f(x)]
[g(x)],None,None,None,None,[f(x)],None,None,None,None
None,None,None,None,None,None,None,None,None,None
None,None,[f(x),g(x)],[f(x),g(x)],[f(x)],None,None,None,None,None
None,None,None,None,None,None,None,None,[f(x)],None
[f(x),g(x),h(x)],None,[f(x),g(x)],[f(x),g(x)],None,None,None,None,None,None
None,None,None,None,None,None,None,None,None,None
None,None,None,None,None,None,None,None,None,[f(x)]
None,None,None,None,[f(x)],[f(x)],[f(x)],None,None,None
None,None,None,None,None,None,None,None,None,None,

None,None,None,None,None,None,None,None,None,[f(x)],
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,[f(x)],None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,

[f(x)],None,None,None,None,None,None,None,None,None,
None,[g(x)],None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,[f(x)],
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,[g(x)],
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,[f(x),g(x),h(x)],None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,

None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
None,None,None,None,None,None,None,None,None,None,
]


@pytest.mark.timeout(60)
def func(i):
    print("")
    print("kamke number ",i)
    # kamke ODEs start at 1 but a list start at 0 
    print("kamke ode = ",kamke1_1[i])
    kamkesol = dsolve(kamke1_1[i],y)
    print("kamkesol = ",kamkesol)
    return(kamkesol)
    

# switch ODEs on or off
@pytest.mark.parametrize("kamkenumber",[  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,                                                
                                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30,                                                
                                         31, 32, 33, 34, 35, 36, 37, 38, 39, 40,                                                
                                         41, 42, 43, 44, 45, 46,         49,                                                

]) 

# check which are expected to fail


## @pytest.mark.skipif(kamkenumber ==, reason="ODE too general")
def test_eval(kamkenumber):
    print("kamkenumber = ",kamkenumber)
    odesol = dsolve(kamke1_1[kamkenumber],y)
    print("odesol = ",odesol)
    #expected = solution_kamke1[kamkenumber]
    #print("odeexpected = ",expected)
    
    #assert odesol == expected
    assert checkodesol(kamke1_1[kamkenumber],odesol)[0]

