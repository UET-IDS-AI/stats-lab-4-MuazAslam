"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation

    RETURN

        analytic_gt5
        analytic_lt5
        analytic_interval
        simulated_gt5
    """

     # STEP 1: Analytical probabilities
    analytic_gt5 = np.exp(-5)
    analytic_lt5 = 1 - np.exp(-5)
    analytic_interval = np.exp(-3) - np.exp(-7)

    # STEP 2: Simulation
    samples = np.random.exponential(scale=1, size=100000)

    # STEP 3: Estimate P(X > 5)
    simulated_gt5 = np.mean(samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5

    raise NotImplementedError


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0

    STEP 1
    Verify non-negativity

    STEP 2
    Compute

        integral_0^∞ f(x) dx

    STEP 3
    Determine if valid PDF

    STEP 4
    Plot f(x) on [0,3]

    RETURN

        integral_value
        is_valid_pdf
    """

    # define function
    f = lambda x: 2*x*np.exp(-x**2)

    # STEP 1 (implicit since formula ≥0 for x≥0)

    # STEP 2 compute integral
    integral_value, _ = quad(f, 0, np.inf)

    # STEP 3 check if valid pdf
    is_valid_pdf = bool(np.isclose(integral_value, 1))

    # STEP 4 plot
    x = np.linspace(0, 3, 400)
    y = f(x)

    plt.plot(x, y)
    plt.title("PDF Candidate: f(x)=2x e^{-x^2}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    return integral_value, is_valid_pdf

    raise NotImplementedError


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)

    STEP 1
    Compute analytically

        P(X > 5)
        P(1 < X < 3)

    STEP 2
    Simulate 100000 samples

    STEP 3
    Estimate probabilities using simulation

    RETURN

        analytic_gt5
        analytic_interval
        simulated_gt5
        simulated_interval
    """

    # STEP 1: analytic probabilities
    analytic_gt5 = np.exp(-5)
    analytic_interval = np.exp(-1) - np.exp(-3)

    # STEP 2: simulation
    samples = np.random.exponential(scale=1, size=100000)

    # STEP 3: simulated estimates
    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval

    raise NotImplementedError


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)

    STEP 1
    Standardize variable

        Z = (X - 10)/2

    STEP 2
    Compute analytically

        P(X ≤ 12)
        P(8 < X < 12)

    STEP 3
    Simulate 100000 samples

    STEP 4
    Estimate probabilities

    RETURN

        analytic_le12
        analytic_interval
        simulated_le12
        simulated_interval
    """

     # STEP 2 analytic probabilities
    analytic_le12 = norm.cdf(12, loc=10, scale=2)
    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # STEP 3 simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    # STEP 4 simulated probabilities
    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval

    raise NotImplementedError
