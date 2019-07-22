# pytorch-hessianfree
PyTorch implementation of Hessian Free optimisation

Implemented some parts of [Training Deep and Recurrent Networks with Hessian-Free Optimization](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_27) by Martens and Sutskever (2012):
* Preconditioner for CG, includes empirical Fisher diagonal (Section 20.11)
* Gauss-Newton matrix and Hessian matrix (Section 20.5 & 20.6)
* Martens' CG stopping criteria (Section 20.4)
* CG backtracking (Section 20.8.7)
* Tikhonov damping with Levenberg-Marquardt like heuristic (Section 20.8.1 & 20.8.5)
* Line-searching (Section 20.8.5)
* Different batches for calculating curvature and gradient, via callable vector b (A x = b) (Section 20.12)

Still yet to do:
* Scale-Sensitive damping (Section 20.8.3)

------------
**Not fully tested, use with caution!**
