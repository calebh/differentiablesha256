# Differentiable SHA256
A fully differentiable implementation of SHA256. Unfortunately suffers from gradient vanishing.

Update: an alternative implementation without the gradient vanishing issue was implemented (see adder branch). The gradients are no longer zero, however the surface of the output hash is rife with local minima, likely because of the avalanche effect. It appears that SHA256 is secure from gradient descent pre-image attacks.

# Dependencies
* PyTorch
