### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="BenchmarkTools", version="1"),
        Pkg.PackageSpec(name="CSV", version="0.9"),
        Pkg.PackageSpec(name="DataFrames", version="1"),
        Pkg.PackageSpec(name="Distributions", version="0.25"),
        Pkg.PackageSpec(name="KernelDensity", version="0.6"),
        Pkg.PackageSpec(name="Plots", version="1"),
        Pkg.PackageSpec(name="PlutoUI", version="0.7"),
        Pkg.PackageSpec(name="StatsBase", version="0.33"),
        Pkg.PackageSpec(name="UrlDownload", version="1"),
    ])
    using BenchmarkTools, CSV, DataFrames, Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, Random, SparseArrays, StatsBase, Statistics, UrlDownload
end

# ╔═╡ 09a9d9f9-fa1a-4192-95cc-81314582488b
html"""
<div style="
position: absolute;
width: calc(100% - 30px);
border: 50vw solid #2e3440;
border-top: 200px solid #2e3440;
border-bottom: none;
box-sizing: content-box;
left: calc(-50vw + 15px);
top: -100px;
height: 100px;
pointer-events: none;
"></div>

<div style="
height: 200px;
width: 100%;
background: #2e3440;
color: #fff;
padding-top: 50px;
">
<span style="
font-family: Vollkorn, serif;
font-weight: 700;
font-feature-settings: 'lnum', 'pnum';
"> <p style="
font-size: 1.5rem;
opacity: 0.8;
">ATS 872: Lecture 6</p>
<p style="text-align: center; font-size: 1.8rem;">
 Bayesian vector autoregression (BVAR)
</p>

<style>
body {
overflow-x: hidden;
}
</style>"""

# ╔═╡ 41eb90d1-9262-42b1-9eb2-d7aa6583da17
html"""
<style>
  main {
    max-width: 800px;
  }
</style>
"""

# ╔═╡ aa69729a-0b08-4299-a14c-c9eb2eb65d5c
md" # Introduction "

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In this session we will move toward understanding Bayesian vector autoregression models."

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ if you want."

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Vector Autoregressions "

# ╔═╡ 41221332-0a22-43c7-b1b2-8a8d84e61dd4
md"""

Vector autoregressions (VARs) have been widely used for macroeconomic forecasting and structural analysis since the seminal work of Sims $(1980)$. In particular, VARs are often served as the benchmark for comparing forecast performance of new models and methods. VARs are also used to better understand the interactions between macroeconomic variables, often through the estimation of impulse response functions that characterize the effects of a variety of structural shocks on key economic variables.

Despite the empirical success of the standard constant-coefficient and homoscedastic VAR, there is a lot of recent work in extending these conventional VARs to models with time-varying regression coefficients and stochastic volatility. These extensions are motivated by the widely observed structural instabilities and time-varying volatility in a variety of macroeconomic time series.

In some of the future lectures we will study a few of these more flexible VARs, including the timevarying parameter (TVP) VAR and VARs with stochastic volatility. An excellent review paper that covers many of the same topics is Koop and Korobilis (2010). We will, however, begin with a basic VAR.

"""

# ╔═╡ 77fde8ac-f019-4b4c-9d03-9a9c0d7ca2a0
md"""

### Basic (reduced form) VAR
"""

# ╔═╡ 7e0462d4-a307-465b-ae27-737431ecb565
md"""

Suppose $\mathbf{y}_{t}=\left(y_{1 t}, \ldots, y_{n t}\right)^{\prime}$ is a vector of dependent variables at time $t$. Consider the following $\operatorname{VAR}(p)$ :

$$\mathbf{y}_{t}=\mathbf{b}+\mathbf{A}_{1} \mathbf{y}_{t-1}+\cdots+\mathbf{A}_{p} \mathbf{y}_{t-p}+\boldsymbol{\varepsilon}_{t}$$

where $\mathbf{b} = \left(b_{1}, \ldots, b_{n}\right)^{\prime}$ is an $n \times 1$ vector of intercepts , $\mathbf{A}_{1}, \ldots, \mathbf{A}_{p}$ are $n \times n$ matrices of autoregressive coefficients and $\boldsymbol{\varepsilon}_{t} = \left(\varepsilon_{1 t}, \ldots, \varepsilon_{n t}\right)^{\prime} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$. 

In other words, $\operatorname{VAR}(p)$ is simply a multiple-equation regression where the regressors are the lagged dependent variables.

Each equation in this system has $k = np + 1$ regressors, and the total system has $nk = n(np + 1)$ coefficients.

To fix ideas, consider a simple example with $n=2$ variables and $p=1$ lag. Then the equation above can be written explicitly as:

$$\underset{\mathbf{y}_t}{\underbrace{\left(\begin{array}{l}
y_{1 t} \\
y_{2 t}
\end{array}\right)}}=\underset{\mathbf{b}}{\underbrace{\left(\begin{array}{l}
b_{1} \\
b_{2}
\end{array}\right)}}
+
\underset{\mathbf{A}}{\underbrace{\left(\begin{array}{ll}
A_{1,11} & A_{1,12} \\
A_{1,21} & A_{1,22}
\end{array}\right)}}
\underset{\mathbf{y}_{t-1}}{\underbrace{\left(\begin{array}{l}
y_{1,t-1} \\
y_{2,t-1}
\end{array}\right)}}
+
\underset{\boldsymbol{\varepsilon}_t}{\underbrace{\left(\begin{array}{c}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right)}}$$

where

$$\left(\begin{array}{l}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right) \sim \mathcal{N}\left(\left(\begin{array}{l}
0 \\
0
\end{array}\right),\left(\begin{array}{ll}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{array}\right)\right)$$

The model runs from $t=1, \ldots, T$, and it depends on the $p$ initial conditions $\mathbf{y}_{-p+1}, \ldots, \mathbf{y}_{0}$. In principle these initial conditions can be modeled explicitly. Here all the analysis is done conditioned on these initial conditions. If the series is sufficiently long (e.g., $T>50)$, both approaches typically give essentially the same results.

"""


# ╔═╡ a34d8b43-152c-42ff-ae2c-1d439c538c8a
md""" ### Link to linear regression """

# ╔═╡ 01213f94-8dee-4475-b307-e8b18806d453
md"""

To derive the likelihood for the $\operatorname{VAR}(p)$, we aim to write the system as the linear regression model

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

Then, we can simply apply the linear regression results to derive the likelihood.

Let's first work out our example with $n=2$ variables and $p=1$ lag. To that end, we stack the coefficients equation by equation, i.e., $\boldsymbol{\beta}=\left(b_{1}, A_{1,11}, A_{1,12}, b_{2}, A_{1,21}, A_{1,22}\right)^{\prime}$. 

Equivalently, we can write it using the `vec` operator that vectorizes a matrix by its columns: $\boldsymbol{\beta}=\operatorname{vec}\left(\left[\mathbf{b}, \mathbf{A}_{1}\right]^{\prime}\right)$. Given our definition of $\boldsymbol{\beta}$, we can easily work out the corresponding regression matrix $\mathbf{X}_{t}$ :

$$\left(\begin{array}{l}
y_{1 t} \\
y_{2 t}
\end{array}\right)=\left(\begin{array}{cccccc}
1 & y_{1(t-1)} & y_{2(t-1)} & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & y_{1(t-1)} & y_{2(t-1)}
\end{array}\right)\left(\begin{array}{c}
b_{1} \\
A_{1,11} \\
A_{1,12} \\
b_{2} \\
A_{1,21} \\
A_{1,22}
\end{array}\right)+\left(\begin{array}{c}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right)$$

Or

$$\mathbf{y}_{t}=\left(\mathbf{I}_{2} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}\right]\right) \boldsymbol{\beta}+\varepsilon_{t}$$

where $\otimes$ is the Kronecker product. More generally, we can write the $\operatorname{VAR}(p)$ as:

$$\mathbf{y}_{t}=\mathbf{X}_{t} \boldsymbol{\beta}+\varepsilon_{t}$$

where $\mathbf{X}_{t}=\mathbf{I}_{n} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}, \ldots, \mathbf{y}_{t-p}^{\prime}\right]$ and $\boldsymbol{\beta}=\operatorname{vec}\left(\left[\mathbf{b}, \mathbf{A}_{1}, \cdots \mathbf{A}_{p}\right]^{\prime}\right) .$ Then, stack the observations over $t=1, \ldots, T$ to get

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

where $\varepsilon \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)$.



"""

# ╔═╡ 868019c7-eeed-449f-9343-a541f6daefe5
md"""

Before we continue, let us show what Kronecker products and `vec` operators actually do in relation to predefined matrices. This might help make the mathematical operations easier to internalise. 

"""

# ╔═╡ 17c1a647-7ebc-4703-bed8-148d3a35ac1d
md"""

### Kronecker product and `vec` operator 

"""

# ╔═╡ 138d3f7c-4575-4232-97e5-5d78ee64ffb1
md""" To create our matrix $\mathbf{X_t}$ we need to define block diagonal matrices with the same matrix repeated a number of times along the diagonal. One of the easiest ways to do this is via the Kronecker product of the two matrices. Let us consider an example for the Kronecker product. We start with a 2x2 identity matrix. """

# ╔═╡ ec2cb83a-39b9-4e4a-a939-85ff25f0c4ac
Id2 = I(2)

# ╔═╡ 13fa6dcc-26c7-4dcb-ba91-532549063191
md""" Now we need to generate a 2x2 matrix to match our example from above. """

# ╔═╡ e6248785-88ae-43ad-967e-924b0024bfb6
z = randn(1, 3)

# ╔═╡ 20c1c904-1a06-48da-b77e-b6a97047163b
md""" Now we create a block diagonal matrix with $\mathbf{z} = \left[1, \mathbf{y}_{t-1}^{\prime}\right]$ repeated two times on the diagonal. """

# ╔═╡ 56ff8382-f0c1-472c-bf85-b34360823a04
kron(Id2, z)

# ╔═╡ 67f2cc01-074c-410b-859c-b79f670cc74d
md""" Now that we have some idea what the Kronecker product does, let us move to the `vec` operator. In this case we are just stacking coefficients. Consider an example with a $\mathbf{b}$ vector and $\mathbf{A_1}$ matrix (as above). """

# ╔═╡ 0244ce0e-42e7-4013-bedd-9e618565d43e
b = randn(1, 2)

# ╔═╡ fa2e2cc2-d103-48d2-88b1-4f62df4f0f3f
A1 = randn(2, 2)

# ╔═╡ c9305189-815f-4e01-afda-aab9c555e9d9
[b; A1]

# ╔═╡ 2e388e43-9fc7-4dd1-8856-0c734ce22cf3
vec([b; A1])

# ╔═╡ 07eeab9e-dd7b-4ff2-9b0f-6f0d5d9a60ec
md""" Hopefully these examples provide some intuition. """

# ╔═╡ 538be57f-54e9-45dd-95f5-12e7a3df51a7
md""" ###  Likelihood contd. """

# ╔═╡ 057a49fa-a68a-4bf0-8b70-7ccd5b8d7931
md"""

Let us now continue with our calculation. Since, 

$$(\mathbf{y} \mid \boldsymbol{\beta}, \mathbf{\Sigma}) \sim \mathcal{N}\left(\mathbf{X} \boldsymbol{\beta}, \mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)$$

the likelihood function is given by:

$$\begin{aligned}
p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma}) &=\left|2 \pi\left(\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)\right|^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}\right)^{-1}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})} \\
&=(2 \pi)^{-\frac{T n}{2}}|\mathbf{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}\left(\mathbf{I}_{T} \otimes \Sigma^{-1}\right)(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})}
\end{aligned}$$

where the second equality holds because $\left|\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right|=|\boldsymbol{\Sigma}|^{T}$ and $\left(\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)^{-1}=\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}^{-1}$. Note that the likelihood can also be written as

$$p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma})=(2 \pi)^{-\frac{T n}{2}}|\mathbf{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2} \sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)}$$

"""

# ╔═╡ 3efef390-b791-40dd-a950-26dcc84c3485
md""" ### Independent priors """

# ╔═╡ 349c3025-3b88-4e7e-9e31-574943f31dc6
md"""

Recall that in the normal linear regression model, we assume independent normal and inverse-gamma priors for the coefficients $\boldsymbol{\beta}$ and the variance $\sigma^{2}$, respectively. Both are conjugate priors and the model can be easily estimated using the Gibbs sampler.

Here a similar result applies. But instead of an inverse-gamma prior, we need a multivariate generalization for the covariance matrix $\boldsymbol{\Sigma}$.

An $m \times m$ random matrix $\mathbf{Z}$ is said to have an inverse-Wishart distribution with shape parameter $\alpha>0$ and scale matrix $\mathbf{W}$ if its density function is given by

$$f(\mathbf{Z} ; \alpha, \mathbf{W})=\frac{|\mathbf{W}|^{\alpha / 2}}{2^{m \alpha / 2} \Gamma_{m}(\alpha / 2)}|\mathbf{Z}|^{-\frac{\alpha+m+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{W} \mathbf{Z}^{-1}\right)}$$

where $\Gamma_{m}$ is the multivariate gamma function and $\operatorname{tr}(\cdot)$ is the trace function. We write $\mathbf{Z} \sim \mathcal{I} W(\alpha, \mathbf{W})$. For $\alpha>m+1, \mathbb{E} \mathbf{Z}=\mathbf{W} /(\alpha-m-1)$.

For the $\operatorname{VAR}(p)$ with parameters $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we consider the independent priors:

$$\boldsymbol{\beta} \sim \mathcal{N}\left(\boldsymbol{\beta}_{0}, \mathbf{V}_{\boldsymbol{\beta}}\right), \quad \boldsymbol{\Sigma} \sim \mathcal{I} W\left(\nu_{0}, \mathbf{S}_{0}\right)$$

"""

# ╔═╡ 5e825c74-431e-4055-a864-d2b366e8ae11
md""" ### Gibbs sampler """

# ╔═╡ 6d451af1-2288-4ee1-a37c-432c14888e16
md"""

Now, we derive a Gibbs sampler for the $\operatorname{VAR}(p)$ with likelihood and priors given in the previous section. Specifically, we derive the two conditional densities $p(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{\Sigma})$ and $p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta})$.

**Step 1: Sample $\boldsymbol{\beta}$**

The first step is easy, as standard linear regression results would apply. 

$$\begin{align}
p\left(\boldsymbol{\beta}\mid\mathbf{y},\boldsymbol{\Sigma}\right) &\propto p\left(\mathbf{y} \mid\boldsymbol{\beta},\boldsymbol{\Sigma}\right)p\left(\boldsymbol{\beta}\right)\\
&\propto\exp\left(-\frac{1}{2}\left(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\right)'(\mathbf{I}_n\otimes\boldsymbol{\Sigma}^{-1})\left(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\right)\right)\exp\left(-\frac{1}{2}\left(\boldsymbol{\beta}-\boldsymbol{\beta}_0\right)'\mathbf{V}_{\boldsymbol{\beta}}^{-1}\left(\boldsymbol{\beta}-\boldsymbol{\beta}_0\right)\right)\\
&\propto\exp\left(-\frac{1}{2}\left(\boldsymbol{\beta}'\mathbf{X}'(\mathbf{I}_n\otimes\boldsymbol{\Sigma}^{-1})\mathbf{X}\boldsymbol{\beta}-2\boldsymbol{\beta}'\mathbf{X}'\mathbf{y}\right)\right)\left(-\frac{1}{2}\left(\boldsymbol{\beta}'\mathbf{V}_{\boldsymbol{\beta}}^{-1}\boldsymbol{\beta}-2\boldsymbol{\beta}'\mathbf{V}_{\boldsymbol{\beta}}^{-1}\boldsymbol{\beta}_0\right)\right)\\
&\propto\exp\left(-\frac{1}{2}\left(\boldsymbol{\beta}'\left(\mathbf{X}'(\mathbf{I}_n\otimes\boldsymbol{\Sigma}^{-1})\mathbf{X}+\mathbf{V}_{\boldsymbol{\beta}}^{-1}\right)\boldsymbol{\beta}-2\boldsymbol{\beta}\left(\mathbf{X}'(\mathbf{I}_n\otimes\boldsymbol{\Sigma}^{-1})\mathbf{y}+\mathbf{V}_{\boldsymbol{\beta}}^{-1}\boldsymbol{\beta}_0\right)\right)\right)
\end{align}$$

The final line is the kernel of a multivariate normal density which implies that

$$(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{\Sigma}) \sim \mathcal{N}\left(\widehat{\boldsymbol{\beta}}, \mathbf{K}_{\boldsymbol{\beta}}^{-1}\right)$$

where

$$\mathbf{K}_{\boldsymbol{\beta}}=\mathbf{V}_{\boldsymbol{\beta}}^{-1}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{X}, \quad \widehat{\boldsymbol{\beta}}=\mathbf{K}_{\boldsymbol{\beta}}^{-1}\left(\mathbf{V}_{\boldsymbol{\beta}}^{-1} \boldsymbol{\beta}_{0}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{y}\right)$$

and we have used the result $\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}\right)^{-1}=\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}$.

**Step 2: Sample $\boldsymbol{\Sigma}$**

Next, we derive the conditional density $p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta})$. Recall that for conformable matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$, we have

$$\operatorname{tr}(\mathbf{A B C})=\operatorname{tr}(\mathbf{B C A})=\operatorname{tr}(\mathbf{C A B})$$

Now, combining the likelihood and the prior, we obtain

$$\begin{aligned}
p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}) & \propto p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma}) p(\boldsymbol{\Sigma}) \\
& \propto|\boldsymbol{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2} \sum_{t=1}^{T}\left(\mathbf{y} t-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)} \times|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{S}_{0} \Sigma^{-1}\right)} \\
& \propto|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+T+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{S}_{0} \Sigma^{-1}\right)} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left[\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \mathbf{\Sigma}^{-1}\right]} \\
&\left.\propto|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+T+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left[\left(\mathbf{S}_{0}+\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime}\right) \Sigma^{-1}\right.}\right]
\end{aligned}$$

which is the kernel of an inverse-Wishart density. In fact, we have

$$(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}) \sim \mathcal{I} W\left(\nu_{0}+T, \mathbf{S}_{0}+\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime}\right)$$

"""

# ╔═╡ 71cfd870-a058-4716-aada-e96c8c968908
md""" ### General remarks on computation """

# ╔═╡ 70313baf-6357-403d-abdd-98e41910aa91
md""" 

Below are some of the general comments from Jaime Cross on computation. If you want to look at computational aspects then these points are worth considering. 

When selecting the hyperparameter values for the prior distributions of the VAR coefficients in $\boldsymbol{\beta}$ note that the same advice in the normal linear regression model applies here. In addition to this, when selecting the hyperparameter values for the error covariance prior, note that
1. The degree of freedom parameter $\nu_0$ must be set such that $\nu_0>n+1$ for mean to exist and $\nu_0>n+3$ for variance to exist. Smaller values result in a larger variance and vice versa. 
2. The scale matrix $\mathbf{S}_0$ can be chosen to pin down the prior mean of $\boldsymbol{\Sigma}$ using the fact that $\mathbb{E}[\boldsymbol{\Sigma}]=\mathbf{S}_0(\nu_0-n-1)^{-1}$, i.e $\mathbf{S}_0=\mathbb{E}[\boldsymbol{\Sigma}](\nu_0-n-1)$. 

In simulation studies, we find that setting $\nu_0 = n + 4$ and $\mathbf{S}_0=\mathbf{I}_n(\nu_0-n-1)$ (i.e. $\mathbb{E}[\boldsymbol{\Sigma}]=\mathbf{I}_n$) works well.

The covariance matrix in the linear regression model is a  $k\times k$ matrix, however in the VAR model it's an $nk\times nk$ matrix. This means that computing its Cholesky factor is relatively more time-consuming, and sampling $\boldsymbol{\beta}$ is more computationally demanding. 

To speed up the computations, we use the fact that the precision matrix $\mathbf{K}_{\boldsymbol{\beta}}^{-1}$ is a [band matrix](https://en.wikipedia.org/wiki/Band_matrix) to efficiently sample $\boldsymbol{\beta}$ using the *precision sampler*

**Precision Sampler for the Multivariate Normal Distribution**: To obtain samples from an $n-$dimensional random vector $\mathbf{V}={V_1,\dots,V_n}$ with a multivariate Normal Distribution given by $N(\boldsymbol{\mu},\boldsymbol{\Sigma}^{-1})$, we perform the following steps
1. Compute the lower Cholesky factorization $\boldsymbol{\Sigma}^{-1}=\mathbf{P}\mathbf{P}'$
2. Generate $\mathbf{Z}=(Z_1,\dots,Z_n)$ by sampling $Z_i\sim N(0,1)$, $i=1,\dots,n$
3. Return the affine transformation $\mathbf{V}=\boldsymbol{\mu} + (\mathbf{P}')^{-1}\mathbf{Z}$
If we want a sample of $R$ draws, then repeat the algorithm $R$ times.

**Exercise**: Use $\mathbf{V}=\boldsymbol{\mu} + (\mathbf{P}')^{-1}\mathbf{Z}$ to show that $\mathbf{V}\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}^{-1})$.

**Remarks**:
1. This is why we specify the posterior distribution for $\boldsymbol{\beta}$ using the precision matrix instead of the usual covariance matrix $\mathbf{K}_{\boldsymbol{\beta}}$.
2. The algorithm is almost the same as the one we used to directly sample from the multivariate normal distribution when estimating the normal linear regression model. The only differences are that we (1) take the Cholesky factor of the precision matrix instead of the covariance matrix, and (2) use the inverse of the conjugate transpose of the lower Cholesky factor $\mathbf{P}$, i.e. $(\mathbf{P}')^{-1}$ in step 3 (as opposed to the lower Cholesky factor $\mathbf{P}$).

"""

# ╔═╡ 2b30e958-6134-4c8c-8dbf-54cf5babef88
md""" ### Gibbs Sampler for the $\operatorname{VAR}(p)$ """

# ╔═╡ 1e22e482-c467-4a57-bf19-96361e2896e6
md"""

We summarize the Gibbs sampler as follows

Pick some initial values $\boldsymbol{\beta}^{(0)}=\mathbf{c}_{0}$ and $\boldsymbol{\Sigma}^{(0)}=\mathbf{C}_{0}>0 .$ Then, repeat the following steps from $r=1$ to $R$

1. Draw $\boldsymbol{\beta}^{(r)} \sim p\left(\boldsymbol{\beta} \mid \mathbf{y}, \boldsymbol{\Sigma}^{(r-1)}\right)$ (multivariate normal).

2. Draw $\boldsymbol{\Sigma}^{(r)} \sim p\left(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}^{(r)}\right)$ (inverse-Wishart).

"""

# ╔═╡ 193f5509-0733-4409-9b88-1d2bc68e3aee
md""" ### Empirical example: Small model of SA economy """

# ╔═╡ c8e7e82f-415e-4b9d-bd62-47c1c60d0741
md"""

In this empirical example we estimate a 3-variable VAR(2) using SA quarterly data on CPI inflation rate, unemployment rate and repo rate from $1994 \mathrm{Q} 1$ to $2020 \mathrm{Q} 4$. These three variables are commonly used in forecasting (e.g., Banbura, Giannone and Reichlin, 2010; Koop and Korobilis, 2010; Koop, 2013) and small DSGE models (e.g., An and Schorfheide, 2007 ).

Following Primiceri (2005), we order the interest rate last and treat it as the monetary policy instrument. The identified monetary policy shocks are interpreted as "non-systematic policy actions" that capture both policy mistakes and interest rate movements that are responses to variables other than inflation and unemployment.

We first implement the Gibbs sampler. Then, given the posterior draws of $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we compute the impulse-response functions of the three variables to a 100-basis-point monetary policy shock.

"""

# ╔═╡ 7557b795-839c-41bf-9586-7b2b3972b28d
begin
	# Parameters
	p = 4            # Number of lags (VAR order)
	burnin = 5000      # Burnin for Gibbs sampler
	nsim = burnin + 40000  # Number of simulation in Gibbs sampler
	n_hz = 50        # Horizon for the IRF
end;

# ╔═╡ 8b97f4dd-31cf-4a55-a72b-66775d79445c
md""" After specifying the parameters for the model, we load the data that we are going to be working with."""

# ╔═╡ 5592c6a8-defc-4b6e-a303-813bdbacaffe
begin
	# Load the dataset and transform to array / matrix
	df = urldownload("https://raw.githubusercontent.com/DawievLill/ATS-872/main/data/sa-data.csv") |> DataFrame 
end

# ╔═╡ 7b8a3ca2-dbdd-4ddf-bed6-eb649796a9b8
md""" We always like to inspect the data, so we plot it to see if there are potential non-stationarities. """

# ╔═╡ e257ca34-c4e3-4a8a-a4dd-cee0d5580347
begin
	plot(df.inf, label = "inflation", color = :steelblue, style = :dash, lw = 1.5)
	plot!(df.gdp, label = "GDP growth", color = :red, style = :dashdot, lw = 1.5)
	plot!(df.repo, label = "repo rate", lw = 1.5, legend = :bottomleft)
end

# ╔═╡ 6c697f4b-c928-43c2-9490-27b0f195857c
data = Matrix(df[:, 1:end]);

# ╔═╡ 8d0a4b08-5cbf-43a8-91f3-5f448893a4c6
md""" In order to estimate the model we use the following function to construct to regression matrix $\boldsymbol{X}$. Recall that $\mathbf{X}_{t}=\mathbf{I}_{n} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}, \ldots, \mathbf{y}_{t-p}^{\prime}\right]$ and we stack $\mathbf{X}_{t}$ over $t=1, \ldots, T$ to obtain $\mathbf{X}$."""

# ╔═╡ 9e3795cf-8695-46c1-9857-728d765caa02
# SUR representation of the VAR(p)
function SUR_form(X, n)

    repX = kron(X, ones(n, 1))
    r, c = size(X)
    idi = kron((1:r * n), ones(c, 1))
    idj = repeat((1:n * c), r, 1)

    # Some prep for the out return value.
    d = reshape(repX', n * r * c, 1)
    out = sparse(idi[:, 1], idj[:, 1], d[:, 1])
end;

# ╔═╡ 171c9c8f-a4a9-4aad-8bab-f8f38d36a94b
md""" Given the posterior draws of $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we then use the function `construct_IR` to compute the impulse-response functions of the three variables to a 100 -basis-point monetary policy shock. More specifically, we consider two alternative paths: in one a 100 -basis-point monetary policy shock hits the system, and in the other this shock is absent. We then let the two systems evolve according to the $\operatorname{VAR}(p)$ written as the regression

$$\mathbf{y}_{t}=\mathbf{X}_{t} \boldsymbol{\beta}_{t}+\mathbf{C} \widetilde{\varepsilon}_{t}, \quad \widetilde{\varepsilon}_{t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{3}\right)$$

for $t=1, \ldots, n_{\mathrm{hz}}$, where $n_{\mathrm{hz}}$ denotes the number of horizons and $\mathbf{C}$ is the Cholesky factor of $\boldsymbol{\Sigma}$. Each impulse-response function is then the difference between these two paths. """

# ╔═╡ 741e914f-4d6d-4249-a324-4dd54fd0f277
function construct_IR(β, Σ, shock)

    n = size(Σ, 1)
    CΣ = cholesky(Σ).L
    tmpZ1 = zeros(p, n)
    tmpZ = zeros(p, n)
    Yt1 = CΣ * shock
    Yt = zeros(n, 1)
    yIR = zeros(n_hz,n)
    yIR[1,:] = Yt1'

    for t = 2:n_hz
        # update the regressors
        tmpZ = [Yt'; tmpZ[1:end-1,:]]
        tmpZ1 = [Yt1'; tmpZ1[1:end-1,:]]

        # evolution of variables if a shock hits
        Random.seed!(12)
        e = CΣ*randn(n,1)
        Z1 = reshape(tmpZ1',1,n*p)
        Xt1 = kron(I(n), [1 Z1])
        Yt1 = Xt1 * β + e

        # evolution of variables if no shocks hit
        Z = reshape(tmpZ',1,n*p)
        Xt = kron(I(n), [1 Z])
        Yt = Xt * β + e

        # the IR is the difference of the two scenarios
        yIR[t,:] = (Yt1 - Yt)'
    end
    return yIR
end;

# ╔═╡ 3bb39875-2f99-4a1f-a7ab-a107b4b95716
md"""

The main function `bvar` is given next. It first loads the dataset, constructs the regression matrix $\mathbf{X}$ using the above function, and then implements the 2 -block Gibbs sampler. Note that within the for-loop we compute the impulse-response functions for each set of posterior draws $(\boldsymbol{\beta}, \boldsymbol{\Sigma})$. Also notice that the shock variable shock is normalized so that the impulse responses are to 100 basis points rather than one standard deviation of the monetary policy shock.

"""

# ╔═╡ 9e949115-a728-48ed-8c06-5cc26b6733bf
function bvar(data)

	# Create y
	data = data[1:end, :] # data matrix
    Y0 = data[1:p, :] # initial conditions
    Y = data[p + 1:end, :] # observations
    T = size(Y, 1)
    n = size(Y, 2)
    y = reshape(Y', T * n, 1)
    k = n * p + 1 # number of coefficients in each equation (21 in total, 3 equations)

    # Specification of the prior (diffuse prior in this case)
	# Prior for β
    β_0 = zeros(n * k) # Best way to initialise with zeros?

	# Prior for Σ
    ν_0 = n + 4
    Σ_0 = I(n) * (ν_0 - n - 1) # sets E[Σ] = I(n)

    # Precision for coefficients and intercepts
    tmp = 10 * ones(k * n) # Precision for coefficients
    tmp[1: p * n + 1 : k * n] = ones(n) # Precision for intercepts
    A = collect(1:k*n)
    Vβ = sparse(A, A, tmp) 

    # Create lagged matrix
    tmpY = [Y0[(end-p+1): end,:]; Y]
    X_til = zeros(T, n * p)

    for i=1:p
        X_til[:, ((i - 1) * n + 1):i * n] = tmpY[(p - i + 1):end - i,:]
    end
    X_til = [ones(T, 1) X_til] # This is the dense X matrix
    X = SUR_form(X_til, n) # Creates a sparse regression array

    # Initialise these arrays (used for storage)
    store_Sig = zeros(nsim, n, n) # For the sigma values
    store_beta = zeros(nsim, k*n) # For the beta values
    store_yIR = zeros(n_hz, n) # For the impulse responses

    # Initialise Markov chain
    β = Array((X'*X) \ (X'*y))
    e = reshape(y - X*β, n, T)
    Σ = e*e'./T # sum of squared residuals

    iΣ = Σ\I(n)
    #iΣ    = Symmetric(iΣ)

    for isim = 1:nsim + burnin

		# Start Gibbs sampling routine
        # Step 1: Sample β from multivariate Normal
        XiΣ = X'*kron(I(T), iΣ)
        XiΣX = XiΣ*X
        Kβ = Vβ + XiΣX
        β_hat = Kβ\(Vβ*β_0 + XiΣ*y)
		
        β = β_hat + (cholesky(Hermitian(Kβ)).L)'\rand(Normal(0,1), n * k) 

        # Step 2: Sample Σ from InverseWishart
        e = reshape(y - X*β, n, T)
        Σ = rand(InverseWishart(ν_0 + T, Σ_0 + e*e'))

        if isim > burnin
            # Store the parameters
            isave = isim - burnin;
            store_beta[isave,:] = β';
            store_Sig[isave,:,:] = Σ;

            # Compute impulse-responses
            CΣ = cholesky(Σ).L

            # 100 basis pts rather than 1 std. dev.
            shock_d = [0; 0; 1]/Σ[n,n] # standard deviation shock
            shock = [0; 0; 1]/CΣ[n,n] # 100 basis point shock
            yIR = construct_IR(β, Σ, shock)
            store_yIR = store_yIR + yIR
        end
    end
    yIR_hat = store_yIR/nsim
    return yIR_hat, store_beta, store_Sig
end;

# ╔═╡ c21aa46d-08ac-4256-a021-83194bad3a5e
md""" What do you think about the IRFs, do they make sense? """

# ╔═╡ 108d294a-c0f0-4325-860e-3c68fef7f1b5
irf_data = bvar(data)[1];

# ╔═╡ abe69a74-74ff-4cc5-9a93-90bd36c48e8a
begin
	plot(irf_data[:, 1], color = :steelblue, style = :dash, lw = 1.5, label = "inflation") 
	plot!(irf_data[:, 2], color = :red, style = :dashdot, lw = 1.5, label = "GDP") 
	plot!(irf_data[:, 3], lw = 1.5, label = "repo rate") 
end

# ╔═╡ 4e402c6a-4e72-4b3d-999e-39c6c92dae6a
md""" Just as in the previous lectures, it is possible to compute posterior means using Monte Carlo integration. In addition, we can plot the marginal posterior distributions for the different variables of interest. However, we have done this many times and we will not repeat the process here. You can play around with the output of the `bvar` function to determine if you are able generate histograms of the posterior distributions. """

# ╔═╡ 878760b6-0a98-4955-b061-6e56ca83dfbf
md""" Now let us move to an example of a BVAR in R. There are many packages that one could use here. However, I am only going to illustrate one. For your project you could use any package you want. """

# ╔═╡ Cell order:
# ╟─09a9d9f9-fa1a-4192-95cc-81314582488b
# ╟─41eb90d1-9262-42b1-9eb2-d7aa6583da17
# ╟─aa69729a-0b08-4299-a14c-c9eb2eb65d5c
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─41221332-0a22-43c7-b1b2-8a8d84e61dd4
# ╟─77fde8ac-f019-4b4c-9d03-9a9c0d7ca2a0
# ╟─7e0462d4-a307-465b-ae27-737431ecb565
# ╟─a34d8b43-152c-42ff-ae2c-1d439c538c8a
# ╟─01213f94-8dee-4475-b307-e8b18806d453
# ╟─868019c7-eeed-449f-9343-a541f6daefe5
# ╟─17c1a647-7ebc-4703-bed8-148d3a35ac1d
# ╟─138d3f7c-4575-4232-97e5-5d78ee64ffb1
# ╠═ec2cb83a-39b9-4e4a-a939-85ff25f0c4ac
# ╟─13fa6dcc-26c7-4dcb-ba91-532549063191
# ╠═e6248785-88ae-43ad-967e-924b0024bfb6
# ╟─20c1c904-1a06-48da-b77e-b6a97047163b
# ╠═56ff8382-f0c1-472c-bf85-b34360823a04
# ╟─67f2cc01-074c-410b-859c-b79f670cc74d
# ╠═0244ce0e-42e7-4013-bedd-9e618565d43e
# ╠═fa2e2cc2-d103-48d2-88b1-4f62df4f0f3f
# ╠═c9305189-815f-4e01-afda-aab9c555e9d9
# ╠═2e388e43-9fc7-4dd1-8856-0c734ce22cf3
# ╟─07eeab9e-dd7b-4ff2-9b0f-6f0d5d9a60ec
# ╟─538be57f-54e9-45dd-95f5-12e7a3df51a7
# ╟─057a49fa-a68a-4bf0-8b70-7ccd5b8d7931
# ╟─3efef390-b791-40dd-a950-26dcc84c3485
# ╟─349c3025-3b88-4e7e-9e31-574943f31dc6
# ╟─5e825c74-431e-4055-a864-d2b366e8ae11
# ╟─6d451af1-2288-4ee1-a37c-432c14888e16
# ╟─71cfd870-a058-4716-aada-e96c8c968908
# ╟─70313baf-6357-403d-abdd-98e41910aa91
# ╟─2b30e958-6134-4c8c-8dbf-54cf5babef88
# ╟─1e22e482-c467-4a57-bf19-96361e2896e6
# ╟─193f5509-0733-4409-9b88-1d2bc68e3aee
# ╟─c8e7e82f-415e-4b9d-bd62-47c1c60d0741
# ╠═7557b795-839c-41bf-9586-7b2b3972b28d
# ╟─8b97f4dd-31cf-4a55-a72b-66775d79445c
# ╠═5592c6a8-defc-4b6e-a303-813bdbacaffe
# ╟─7b8a3ca2-dbdd-4ddf-bed6-eb649796a9b8
# ╟─e257ca34-c4e3-4a8a-a4dd-cee0d5580347
# ╠═6c697f4b-c928-43c2-9490-27b0f195857c
# ╟─8d0a4b08-5cbf-43a8-91f3-5f448893a4c6
# ╠═9e3795cf-8695-46c1-9857-728d765caa02
# ╟─171c9c8f-a4a9-4aad-8bab-f8f38d36a94b
# ╠═741e914f-4d6d-4249-a324-4dd54fd0f277
# ╟─3bb39875-2f99-4a1f-a7ab-a107b4b95716
# ╠═9e949115-a728-48ed-8c06-5cc26b6733bf
# ╟─c21aa46d-08ac-4256-a021-83194bad3a5e
# ╠═108d294a-c0f0-4325-860e-3c68fef7f1b5
# ╟─abe69a74-74ff-4cc5-9a93-90bd36c48e8a
# ╟─4e402c6a-4e72-4b3d-999e-39c6c92dae6a
# ╟─878760b6-0a98-4955-b061-6e56ca83dfbf
