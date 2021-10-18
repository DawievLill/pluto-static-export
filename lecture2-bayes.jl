### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="Distributions", version="0.25"),
        Pkg.PackageSpec(name="KernelDensity", version="0.6"),
        Pkg.PackageSpec(name="Plots", version="1"),
        Pkg.PackageSpec(name="PlutoUI", version="0.7"),
        Pkg.PackageSpec(name="QuadGK", version="2"),
        Pkg.PackageSpec(name="RCall", version="0.13"),
        Pkg.PackageSpec(name="StatsBase", version="0.33"),
        Pkg.PackageSpec(name="StatsPlots", version="0.14"),
        Pkg.PackageSpec(name="Turing", version="0.18"),
    ])
    using Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, QuadGK, Random, RCall, StatsBase, Statistics, StatsPlots, Turing
end

# ╔═╡ 6f157a69-be96-427b-b844-9e76660c4cd6
using BenchmarkTools

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
">ATS 872: Lecture 2</p>
<p style="text-align: center; font-size: 1.8rem;">
 Introduction to Bayesian econometrics
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
    max-width: 900px;
  }
</style>
"""

# ╔═╡ aa69729a-0b08-4299-a14c-c9eb2eb65d5c
md" # Introduction "

# ╔═╡ bcba487d-0b1f-4f08-9a20-768d50d67d7f
md""" 

> "When the facts change, I change my mind. What do you do, sir?" -- **John Maynard Keynes**

"""

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In this session we will be looking at the basics of Bayesian econometrics / statistics. We will start with a discussion on probability and Bayes' rule and then we will move on to discuss single parameter models. Some math will be interlaced with the code. I assume some familiarity with linear algebra, probability and calculus for this module. The section is on probability is simply a high level overview that leads us to our derivation of Bayes' theorem / rule. "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ to show support."

# ╔═╡ 4666fae5-b072-485f-bb3a-d28d54fb5274
md" ## Basic probability (preliminaries) "

# ╔═╡ ec9e8512-85fb-4d23-a4ad-7c758cd87b62
md"
The probability of an **event** is a real number between $0$ and $1$. Here $0$ indicates the 'impossible event' and $1$ the certain event. 

> **Note:** Probability $0$ does not mean impossible -- see this [cool video](https://www.youtube.com/watch?v=ZA4JkHKZM50) on this topic. Possibility is tied to the idea of probability density, but we won't go into detail on this idea here. If you are interested in this topic you will have to learn a little [measure theory](https://en.wikipedia.org/wiki/Measure_(mathematics)) to fully understand. If you want to learn more about probability theory in a semi-rigorous set of notes, I recommend the following by [Michael Betancourt](https://betanalpha.github.io/writing/). 

We define $A$ as an event and $P(A)$ as the probability of the event $A$. Therefore, 

$\{P(A) \in \mathbb{R}: 0 \leq P(A) \leq 1\}$

This means that the probability of the event to occur is the set of all real numbers between $0$ and $1$ including $0$ and $1$. We have three axioms of probability, namely, 

1. **Non-negativity**: For all $A$, $P(A) \geq 0$
2. **Additivity**: For two mutually exclusive $A$ and $B$, $P(A) = 1 - P(B)$ and $P(B) = 1 - P(A)$
3. **Normalisation**: Probability of all possible events $A_1, A_2, \ldots$ must add up to one, i.e. $\sum_{n \in \mathbb{N}} P (A_n) = 1$

"

# ╔═╡ 14339a8b-4486-4869-9cf2-6a5f1e4363ac
md" With the axioms established, we are able construct all mathematics pertaining to probability. The first topic of interest to us in the Bayesian school of thought is conditional probability. "

# ╔═╡ 2acbce7f-f528-4047-a2e0-0d7c710e37a1
md" ### Conditional probability "

# ╔═╡ f7592ef8-e966-4113-b60e-559902911e65
md" This is the probability that one event will occur if another has occurred or not. We use the notation $P(A | B)$, which can be read as, the probability that we have observed $A$ given that we have already observed $B$."

# ╔═╡ 6d0bb8df-7211-4fca-b021-b1c35da7f7db
md" We can illustrate with an example. Think about a deck of cards, with 52 cards in a deck. The probability that you are dealt a 🂮 is $1/52$. In other words, $P(🂮 )=\left(\frac{1}{52}\right)$, while the probability of begin a dealt a 🂱 is $P(🂱)=\left(\frac{1}{52}\right)$. However, the probability that you will be dealt 🂱 given that you have been dealt 🂮 is

$P(🂱 | 🂮)=\left(\frac{1}{51}\right)$

This is because we have one less card, since you have already been dealt 🂮. Next we consider joint probability. "

# ╔═╡ 4dfcf3a2-873b-42be-9f5e-0c93cf7220fc
md" ### Joint probability "

# ╔═╡ b8a11019-e87e-4ccb-b9dd-f369ba51a182
md" Join probability is the probability that two events will both occur. Let us extend our card problem to all the kings and aces in the deck. Probability that you will receive an Ace ($A$) and King ($K$) as the two starting cards:

$P(A, K) = P(A) \cdot P(K \mid A)$ 

We can obviously calculate this by using Julia as our calculator... "


# ╔═╡ d576514c-88a3-4292-93a8-8d23edefb2e1
begin
	p_a = 1/ 13 # Probabilty of A
	p_kga = 4 / 51 # Probability of K given A
	p_ak = p_a * p_kga # Joint probability
end

# ╔═╡ 3abcd374-cb1b-4aba-bb3d-09e2819bc842
md" One should note that $P(A, K) = P(K, A)$ and that from that we have 

$P(A) \cdot P(K \mid A) =  P(K) \cdot P(A \mid K)$

**NB note**: Joint probability is commutative, but conditional probability is **NOT**. This means that generally $P(A \mid B) \neq P(B \mid A)$. In our example above we have some nice symmetry, but this doesnt occur often.  "

# ╔═╡ 411c06a3-c8f8-4d1d-a247-1f0054701021
md" ### Bayes' Theorem "

# ╔═╡ 46780616-1282-4e6c-92ec-5a965f1fc701
md" From the previous example we now know that for two events $A$ and $B$ the following probability identities hold, 

$\begin{aligned} P(A, B) &=P(B, A) \\ P(A) \cdot P(B \mid A) &=P(B) \cdot P(A \mid B) \end{aligned}$

From this we are ready to derive Bayes' rule. 

$\begin{aligned} P(A) \cdot P(B \mid A) &=\overbrace{P(B)}^{\text {this goes to the left}} P(A \mid B) \\ \frac{P(A) \cdot P(B \mid A)}{P(B)} &=P(A \mid B) \\ P(A \mid B) &=\frac{P(A) \cdot P(B \mid A)}{P(B)}  \end{aligned}$

Bayesian statistics uses this theorem as method to conduct inference on parameters of the model given the observed data. "

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Bayesian thinking "

# ╔═╡ 7bbecc2b-8c49-458c-8f2e-8792efa62a32
md"""

> A good discussion on Bayesian thinking is the one by [Cam Davidson-Pilon](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb). You can read the first part of this chapter to get an idea of what the Bayesian way of approaching probability is all about. 

"""

# ╔═╡ 03288125-5ebd-47d3-9e1a-32e2308dbd51
md" Consider an economic model that describes an AR($1$) process

$\begin{equation*} y_{t}=\mu+\alpha y_{t-1}+\varepsilon_{t}, \quad \varepsilon_{t} \sim \mathcal{N}\left[0, \sigma^{2}\right] \end{equation*}$ 

where $\mu$, $\alpha$ and $\sigma^{2}$ are parameters in a vector $\theta$. In the usual time series econometrics course one would try and estimate these unkown parameters with methods such as maximum likelihood estimation (MLE), as you did in the first part of the course. So we want to estimate $\theta = \{\mu, \alpha, \sigma^{2}\}$ 

Unobserved variables are usually called **parameters** and can be inferred from other variables. $\theta$ represents the unobservable parameter of interest, where $y$ is the observed data. 

Bayesian conclusions about the parameter $\theta$ is made in terms of **probability statements**. Statements are conditional on the observed values of $y$ and can be written $p(\theta \mid y)$: given the data, what do we know about $\theta$? 

**Notation remark**: You will see that we have switched to a small letter $p$ for probability distribution of a random variable. Previously we have used a capital $P$ to relate probability of events. You will often see probability of events written as $\mathbb{P}$ in textbooks as well. 

The Bayesian view is that there may be many possible values for $\theta$ from the population of parameter values $\Theta$. The frequentist view is that only one such a $\theta$ exists and that repeated sampling and greater frequency of observation will reveal this value. In other words, $\theta$ is regarded as a random variable in the Bayesian setting. This means that we have some subjective belief about $\theta$ should be in the Bayesian view and there is uncertainty attached as to what the parameter value actually is. 

Bayesians will test initial assertions regarding $\theta$ using data on $y$ to investigate probability of assertions. This provides probability distribution over possible values for $\theta \in \Theta$. 

For our model we start with a numerical formulation of joint beliefs about $y$ and $\theta$ expressed in terms of probability distributions.

1. For each $\theta \in \Theta$ the prior distribution $p(\theta)$ describes belief about true population characteristics
2. For each $\theta \in \Theta$ and $y \in \mathcal{Y}$, our sampling model $p(y \mid \theta)$ describes belief that $y$ would be the outcome of the study if we knew $\theta$ to be true.

Once data is obtained, the last step is to update beliefs about $\theta$. For each $\theta \in \Theta$ our posterior distribution $p(\theta \mid y)$ describes our belief that $\theta$ is the true value having observed the dataset.

To make probability statements about $\theta$ given $y$, we begin with a model providing a **joint probability distribution** for $\theta$ and $y$. Joint probability density can be written as product of two densities: the prior $p(\theta)$ and sampling distribution $p(y \mid \theta)$

$p(\theta, y) = p(\theta)p(y \mid \theta)$

However, using the properties of conditional probability we can also write the joint probability density as

$p(\theta, y) = p(y)p(\theta \mid y)$

Setting these equations equal and rearranging provides us with Bayes' theorem / rule, as discussed before. 

$p(y)p(\theta \mid y) = p(\theta)p(y \mid \theta) \rightarrow p(\theta \mid y) = \frac{p(y \mid \theta)p(\theta)}{p(y)}$
"

# ╔═╡ eed2582d-1ba9-48a1-90bc-a6a3bca139ba
md" We are then left with the following formulation:

$$\underbrace{p(\theta \mid y)}_{\text {Posterior }}=\frac{\overbrace{p(y \mid \theta)}^{\text {Likelihood }} \cdot \overbrace{p(\theta)}^{\text {Prior }}}{\underbrace{p(y)}_{\text {Normalizing Constant }}}$$ 

We can safely ignore $p(y)$ in Bayes' rule since it does not involve the parameter of interest $(\theta)$, which means we can write 

$p(\theta|y)\propto p(\theta)p(y|\theta)$

The **posterior density** $p(\theta \mid y)$ summarises all we know about $\theta$ after seeing the data, while the **prior density** $p(\theta)$ does not depend on the data (what you know about $\theta$ prior to seeing data). The **likelihood function** $p(y \mid \theta)$ is the data generating process (density of the data conditional on the parameters in the model). "

# ╔═╡ 0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
md" ### Model vs. likelihood "

# ╔═╡ c6d22671-26e8-4ba3-865f-5cd449a6c9be
md" The following is important to point out, since it can create some confusion (at least I found it confusing at first). The sampling model is shown as $p_{Y}(Y \mid \Theta = \theta) = p(y \mid \theta)$ as a function of $y$ given **fixed** $\theta$ and describes the aleatoric (unknowable) uncertainty.  

On the other hand, likelihood is given as $p_{\Theta}(Y=y \mid \Theta) = p(y \mid \theta) =L(\theta \mid y)$ which is a function of $\theta$ given **fixed** $y$ and provides information about epistemic (knowable) uncertainty, but is **not a probability distribution** 

Bayes' rule combines the **likelihood** with **prior** uncertainty $p(\theta)$ and transforms them to updated **posterior** uncertainty."

# ╔═╡ 34b792cf-65cf-447e-a30e-40ebca992427
md"""

#### What is a likelihood? 

"""

# ╔═╡ b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
md"""

This is a question that has bother me a fair bit since I got started with econometrics. So I will try and give an explanation with some code integrated to give a better idea. This has been mostly inspired by the blog post of [Jim Savage](https://khakieconomics.github.io/2018/07/14/What-is-a-likelihood-anyway.html).

In order to properly fit our Bayesian models we need to construction some type of function that lets us know whether the values of the unknown components of the model are good or not. Imagine that we have some realised values and we can form a histogram of those values. Let us quickly construct a dataset. 

"""

# ╔═╡ f364cb91-99c6-4b64-87d0-2cd6e8043142
data_realised = 3.5 .+ 3.5 .* randn(100) # This is a normally distributed dataset that represents our true data. 

# ╔═╡ 6087f997-bcb7-4482-a343-4c7830967e49
histogram(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), bins  = 20)

# ╔═╡ bf5240e8-59fc-4b1f-8b0d-c65c196ab402
md"""

We would like to fit a density function on this histogram so that we can make probabilistic statements about observations that we have not yet observed. Our proposed density should try and match model unknowns (like the location and scale). Let us observe two potential proposed densities. One is bad guess and the other quite good. 

"""

# ╔═╡ 8165a49f-bd0c-4ad6-8631-cae7425ca4a6
begin
	Random.seed!(1234) 
	bad_guess = -1 .+ 1 .* randn(100);
	good_guess = 3 .+ 3 .* randn(100);
end

# ╔═╡ 5439c858-6ff3-4faa-9145-895390240d76
begin
	density(bad_guess, color = :red,lw = 1.5,  fill = (0, 0.3, :red), title = "Bad Guess")
	histogram!(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), norm = true, bins = 20)
end

# ╔═╡ 0504929d-207c-4fb7-a8b9-14e21aa0f74b
begin
	density(good_guess, color = :steelblue,lw = 1.5,  fill = (0, 0.3, :steelblue), title = "Good Guess")
	histogram!(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), norm = true, bins = 20)
end

# ╔═╡ 169fbcea-4e82-4756-9b4f-870bcb92cb93
md"""

A likelihood function would return a higher value for the proposed density in the case of the good guess. There are many functions that we could use to determine wheter proposed model unknowns result in a better fit. Likelihood functions are one particular approach. Like we mentioned before, these likelihood functions represent a data generating process. The assumption is that the proposed generative distribution gave rise to the data in question. We will continue this discussion a bit later on.  

"""

# ╔═╡ 699dd91c-1141-4fb6-88fc-f7f05906d923
md" ## Bernoulli and Binomial "

# ╔═╡ 6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
md" In this section we will be looking at some single parameter models. In other words, models where we only have a single unknown parameter of interest. This will draw on our knowledge from random variables and their distributions in the previous lecture.  "

# ╔═╡ 284d0a23-a329-4ea7-a069-f387e21ba797
md"""

### Bernoulli random variable 

"""

# ╔═╡ bb535c41-48cb-44fd-989b-a6d3e310406f
md"""

Let us give a general description of the Bernoulli and Binomial random variables and their relation to each other. We will start with data that has a Bernoulli distribution. In the section on estimating bias in a coin we will work through a specific example. Here we provide the framework.

Consider an experiment (such as tossing a coin) that is repeated $N$ times. Each time we conduct this experiment / trial we can evaluate the outcome as being a success or failure.

In this case the $y_{i}$'s, for $i = 1, \ldots, N$, are random variables for each repetition of the experiment. A random variable is a function that maps the outcome to a value on the real line. In our example, the realisation of $y_{i}$ can be $0$ or $1$ depending on whether the experiment was a success or failure.  

The probability of success is represented by $\theta$, while the probability of failure is given by $1- \theta$. This is considered an Bernoulli event. Our goal is to gain an estimate for $\theta$.

A binary random variable  $y \in \{0, 1\}$, $0 \leq \theta \leq 1$ follows a Bernoulli distribution if

$p\left(y \mid \theta\right)=\left\{\begin{array}{cl}\theta & \text { if } y=1 \\ 1-\theta & \text { if } y=0\end{array}\right.$

We combine the equations for the probabilities of success and failure into a single expression 

$p(y \mid \theta) = \theta^{y} (1 - \theta)^{1-y}$

We can also figure out the formula for the likelihood of a whole set of outcomes from **multiple flips**. Denote the outcome fo the $i$th flip as $y_{i}$ and denote the set of outcomes as $\{y_{i}\}$. If we assume outcomes are independent of each other then we can derive the following formula for the probability of set of outcomes

$\begin{aligned} p(y \mid \theta) &=\prod_{i=1}^{N} p\left(y_{i} \mid \theta\right) \\ 
  =& \prod_{i}^{N} \theta^{y_i}(1-\theta)^{(1-y_i)} \\
  =& \theta^{\sum_{i} {y_i}}(1-\theta)^{\sum_{i}(1-y_i)} \\
  =& \theta^{y}(1-\theta)^{N - y} 
\end{aligned}$


"""

# ╔═╡ 9e7a673e-7cb6-454d-9c57-b6b4f9181b06
md" Soon we will write some code for this likelihood. It turns out to be the likelihood for a Binomial random variable, which consists of $N$ independent Bernoulli trials."

# ╔═╡ fe8f71b2-4198-4a12-a996-da254d2cc656
md" ### Binomial random variable "

# ╔═╡ 7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
md"""

It is worthwhile mentioning the Binomial distribution at this stage. The Bernoulli distribution represents the success or failure of a **single Bernoulli trial**. The Binomial distribution represents the number of successes and failues in $N$ independent Bernoulli trials for some given value of $N$. 

The probability of several events in independent trials is $\theta \cdot \theta(1-\theta)\cdot\theta(1-\theta)(1-\theta)\ldots$ 

If there are $N$ trials then the probability that a success occurs $y$ times is

$$\begin{align*}
      p(y \mid \theta, N) & = \frac{N!}{y!(N-y)!} \theta^y(1-\theta)^{N-y} \\
      &= \binom{N}{y} \theta^y(1-\theta)^{N-y}
    \end{align*}$$

We can easily use the `Distributions.jl` package discussed in the previous lecture to draw from the distribution, but let us try and see what happens if we try and code this up ourselves. 
"""

# ╔═╡ f45eb380-7b43-4fd0-af34-89ffd126a63f
md" Working with distributions is a big part of Bayesian statistics, so for our first example today we will write a function that generates draws from a **binomial distribution**."

# ╔═╡ 4084f646-bce6-4a21-a529-49c7774f5ad1
md" The binomial random variable $y \sim \text{Bin}(n, p)$ represents the number of successes in $n$ binary trials where each trial succeeds with probability $p$. We are going to use the `rand()` command to write a function called `binomial_rv` that generates one draw of $y$. "

# ╔═╡ 3c49d3e4-3555-4d18-9445-5347247cf639
function binomial_rv(n, p)
	
	count = 0
    for i in 1:n
        if rand(n)[i] < p
            count += 1 # or count = count + 1
        end
    end
    return count
end

# ╔═╡ 0520b5e3-cf92-4990-8a65-baf300b19631
# The equivalent code in R. Code is almost exactly the same for this example

R"binomial_rv_r <- function(n, p) {
  # Write the function body here
  y <- c(1:n)
  count <- 0
  for (i in seq_along(y)) {
    if (runif(n)[i] < p) {
      count <- count + 1
    }
  }
  return(count)
}";

# ╔═╡ 98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
md" Given a value of $n = 10000$ indepedent trials, how many times will we observe success? "

# ╔═╡ f7b158af-537e-4d9f-9c4c-318281097dce
PlutoUI.with_terminal() do
	@time binomial_rv(100000, 0.5) # Compare this with the time it takes to run in R. 
end

# ╔═╡ 2cb41330-7ebd-45de-9aa1-632db6f9a140
R"system.time(a <- binomial_rv_r(100000, 0.5))"

# ╔═╡ 69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
md" Now let us conduct some experiments with our new binomial random variable. "

# ╔═╡ b6da2479-1545-4b1d-8d7f-07d6d1f67635
md"""

!!! note "Interactive sliders for Binomial random variable"
	Shift these sliders around to see what happens to the graph below. Try fixing values for $p$ and increase the number of $N$, what happens to the distribution?
	
``N``: $(@bind binom_n Slider(1:100; show_value=true, default=10))
``p``: $(@bind binom_p Slider(0.01:0.01:0.99; show_value=true, default=0.5))

"""

# ╔═╡ c4cc482b-815b-4747-9f5a-5779d69086f7
Plots.plot(
    Binomial(binom_n, binom_p);
    seriestype=:sticks,
    markershape=:circle,
    xlabel=raw"$k$",
    ylabel=raw"$p_{Y\mid \Theta}(y \mid \theta)$",
    title="\$\\operatorname{Binom}($binom_n, $binom_p)\$",
    label=false,
)

# ╔═╡ 9016cba4-58f0-4b7f-91af-66faaf3fe99c
md" Naturally, one would want to use a pre-packaged solution to sampling with a binomial random variable. The `Distributions.jl` package contains optimised routines that work faster than our code, but is still a good idea to code some things yourself to fully understand the mechanisms. " 

# ╔═╡ 828166f7-1a69-4952-9e3b-50a99a99789f
md" #### Estimating bias in a coin  "

# ╔═╡ 24c4d724-5911-4534-a5c6-3ab86999df43
md"""
For an example lets look at estimating bias in a coin. We observe the number of heads that result from flipping one coin and we estimate its underlying probability of coming up heads. We want to create a descriptive model with meaningful parameters. The outcome of a flip will be given by $y$, with $y=1$ indicating heads and $y = 0$ tails. 

We need underlying probability of heads as value of parameter $\theta$. This can be written as $p(y = 1 \mid \theta) = \theta$. The probability that the outcome is heads, given a parameter value of $\theta$, is the value $\theta$. We also need the probability of tails, which is the complement of probability of heads $p(y = 0 \mid \theta) = 1 - \theta$. 

Combine the equations for the probability of heads and tails, we have the same as before,  

$$\begin{align*}
  p(y \mid \theta)  = \theta^{y}(1-\theta)^{1-y}
\end{align*}$$

We have established this probability distribution is called the Bernoulli distribution. This is a distribution over two discrete values of $y$ for a fixed value of $\theta$. The sum of the probabilities is $1$ (which must be the case for a probability distribution).

$$\begin{align*}
  \sum_{y} p(y \mid \theta) = p(y = 1 \mid \theta) + p(y = 0 \mid \theta) = \theta + (1-\theta) = 1
\end{align*}$$

If we consider $y$ fixed and the value of $\theta$ as variable, then our equation is a **likelihood function** of $\theta$.

This likelihood function is **NOT** a probability distribution! 

Suppose that $y = 1$ then $\int_{0}^{1}\theta^{y}(1-\theta)^{1-y}\text{d}\theta = \int_{0}^{1}\theta^{y}\text{d}\theta = 1/2$

The formula for the probability of the set of outcomes is given by

$$\begin{align*}
  p(y \mid \theta)  =& \prod_{i}^{N} p(y_i \mid \theta)  \\
  =& \prod_{i}^{N} \theta^{y_i}(1-\theta)^{(1-y_i)} \\
  =& \theta^{\sum_{i} {y_i}}(1-\theta)^{\sum_{i}(1-y_i)} \\
  =& \theta^{\#\text{heads}}(1-\theta)^{\#\text{tails}} \\
	=& \theta^y(1-\theta)^{N - y}
\end{align*}$$

Let us quickly talk about this likelihood, so that we have clear vision of what it looks like. We start with an example where there are $5$ coin flips and $1$ of them is heads (as can be seen below). 

"""

# ╔═╡ 5046166d-b6d8-4473-8823-5209aac59c84
begin
	Random.seed!(1237)
	coin_seq = Int.(rand(Bernoulli(0.4), 5))
end

# ╔═╡ 82d0539f-575c-4b98-8679-aefbd11f268e
md"""

Let us say that we think the probability of heads is $0.3$. Our likelihood can be represented as

$L(\theta \mid y) = p(y = (0, 1, 0, 0, 0) \mid \theta) = \prod_{i=1}^{N} \theta^{y_{i}} \times (1 - \theta)^{1 - y_{i}} = \theta ^ y (1- \theta) ^{N - y}$

Do we think that the proposed probability of heads is a good one? We can use the likelihood function to perhaps determine this. We plot the values of the likelihood function for this data evaluated over the possible values that $\theta$ can take. 
"""

# ╔═╡ 00cb5973-3047-4c57-9213-beae8f116113
begin
	grid_θ = range(0, 1, length = 1001) |> collect; # discretised 
	binom(grid_θ, m, N) = (grid_θ .^ m) .* ((1 .- grid_θ) .^ (N - m))
end

# ╔═╡ c0bba3aa-d52c-4192-8eda-32d5d9f49a28
md"""

!!! note "Coin flippling likelihood"
	Below we have the likelihood function for our coin flipping problem. Shift the sliders to see changes in the likelihood. 

heads = $(@bind m Slider(0:50, show_value = true, default=1));
flips = $(@bind N Slider(1:50, show_value = true, default=5)); 

"""

# ╔═╡ 9e3c0e01-8eb6-4078-bc0f-019466afba5e
binomial  = binom(grid_θ, m, N);

# ╔═╡ ab9195d6-792d-4603-8605-228d479226c6
max_index = argmax(binom(grid_θ, m, N)); # Get argument that maximises this function 

# ╔═╡ e42c5e18-a647-4281-8a87-1b3c6c2abd33
likelihood_max = grid_θ[max_index]; # Value at which the likelihood function is maximised. Makes sense, since we have 3 successes in 5 repetitions. 

# ╔═╡ 5d6a485d-90c4-4f76-a27e-497e8e12afd8
begin
	plot(grid_θ, binomial, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood", legend = false)
	plot!([likelihood_max], seriestype = :vline, lw = 2, color = :black, ls = :dash, alpha =0.7, xticks = [likelihood_max])
end

# ╔═╡ 987aeb82-267a-4ca9-bf21-c75a49edad70
bin_area = sum(binomial)

# ╔═╡ 9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
md"""

What do we notice about the likelihood function? 

1. It is **NOT** a probability distribution, since it doesn't integrate to one (we check this with some code above)
2. We notice that in our particular example the function is maximised at $likelihood_max. This maximum point of the likelihood function is known as the maximum likelihood estimate of our parameter given our data. You have dealt with this quantity before. Formally it is, 

$\hat{\theta}_{M L E}=\operatorname{argmax}_{\theta}(p(y \mid \theta))$

This example shows that it can be dangerous to use maximum likelihood with small samples. The true success rate is $0.4$ but our estimate provided a value of $0.2$.

In this case, our prior information (subjective belief) was that the probability of heads should be $0.3$. This could have helped is in this case get to a better estimate, but unfortunately maximimum likelihood does not reflect this prior belief. This means that we are left with a success rate equal to the frequency of occurence. 

"""

# ╔═╡ 36e6f838-8277-480b-b48d-f70e8fe011eb
md"""

The next step then is to establish the prior, which will be an arbitrary choice here. One assumption could be that the factory producing the coins tends to produce mostly fair coins. Indicate number of heads by $y$ and number of flips by $N$. We need to specify some prior, and we will use the Triangular distribution for our prior in the next section.

Let us code up the likelihood, prior and updated posterior for this example. In order to do this let us implement the grid method. There are many other ways to do this. However, this method is easy to do and gives us some good coding practice.

"""

# ╔═╡ 11b8b262-32d2-4620-bc6b-afca4a5ce977
md" #### The grid method "

# ╔═╡ 89f7f633-4f75-4ef5-aa5b-80e318d14ee5
md" There are four basic steps behind the grid method.

1. Discretise the parameter space if it is not already discrete.
2. Compute prior and likelihood at each “grid point” in the (discretised) parameter space.
3. Compute (kernel of) posterior as 'prior $\times$ likelihood' at each “grid point”.
4. Normalize to get posterior, if desired."

# ╔═╡ 11552b20-3407-4d0b-b07d-1488c8e8a759
md" The first step then in the grid method is to create a grid. The parameter is $\theta$ and we will discretise by selecting grid points between $0$ and $1$. For our first example let us choose $1001$ grid points. "

# ╔═╡ 599c2f09-ad5e-4f39-aa7d-c1ba155725d6
coins_grid = range(0, 1, length = 1001) |> collect;

# ╔═╡ 09ec10d9-a604-480d-8e82-59e84a843749
md" Now we will add a triangular prior across the grid points with a peak at $\theta = 0.5$ and plot the resulting graph."

# ╔═╡ 071761f8-a187-47a6-8fee-5fc91e65d04c
m₁ = 1 # Number of heads

# ╔═╡ f001b040-2ae7-4e97-b229-eebaabb537b0
N₁ = 5 # Number of flips

# ╔═╡ 9a2d5bdf-9597-40c7-ac18-bb27f187912d
triangle_prior = TriangularDist(0, 1); # From the Distributions.jl package

# ╔═╡ f6e6c4bf-9b2f-4047-a6cc-4ab9c3ae1420
plot(triangle_prior, coins_grid, xlab = "theta", ylab = "prior", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))

# ╔═╡ 8382e073-433b-4e42-a6fa-d5a051586457
md" In this small dataset we have $1$ success out of $5$ attempts. Our distribution function will calculate the probability that we want for a given value of $\theta$. We want to do this for each value of $\theta$, but using same values for $m$ and $N$ each time.  "

# ╔═╡ 9678396b-d42c-4c7c-821c-08126895efd3
binomial₁ = binom(grid_θ, m₁, N₁);

# ╔═╡ 0a1d46ed-0295-4000-9e30-3ad838552a7e
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))
	plot!(coins_grid, binomial₁, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood")
end

# ╔═╡ e5aade9a-4593-4903-bc3a-3a37f9f71c98
md"""
We can normalise the likelihood for the purpose of plotting. We can do this by dividing by the sum of the likelihoods and by the width of the spaces betwen the grid points.
"""

# ╔═╡ 87db6122-4d28-45bf-b5b0-41189792199d
likelihood_norm = binomial₁ / sum(binomial₁) / 0.001; # Normalised

# ╔═╡ c6e9bb86-dc67-4f42-89da-98581a0c3c98
md" Likelihoods are **NOT** probability mass functions or probability density functions so the total area under the likelihood function is not generally going to be $1$.  "

# ╔═╡ b81924b8-73f6-4b28-899c-ec417d538dd4
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Normalised likelihood")
end

# ╔═╡ c3d2ba03-e676-4f0f-bafd-feecd0e4e414
md" The hardest part is done, now we only need to multiply by the prior and likelihood to get the posterior. "

# ╔═╡ 97c0c719-fb73-4571-9a6c-629a98cc544d
prior = pdf(triangle_prior, coins_grid); # Extract the values for the prior at the grid points

# ╔═╡ 0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
posterior = prior .* likelihood_norm; # Calculate the posterior

# ╔═╡ 4e790ffa-554d-4c46-af68-22ecb461fb7b
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ 4b141ffc-4100-47e3-941a-4e72c784ccf0
md" Play around with the sliders here so that you can see what happens to the posterior once it gets updated with new information from the data. In addition, what happens once the size of the dataset increases? What role does the prior play?"

# ╔═╡ 219aafcb-17b1-4f5f-9c2b-9b713ba78b18
md"""
heads = $(@bind y₂ Slider(1:100, show_value = true, default=1));
flips = $(@bind n₂ Slider(1:100, show_value = true, default=5))
"""

# ╔═╡ 2833e081-45d6-4f64-8d1e-b3a5895b7952
begin
	b₂ = Binomial.(n₂, coins_grid)
	likelihood_2 = pdf.(b₂, y₂)
	likelihood_norm_2 = likelihood_2 / sum(likelihood_2) / 0.001
	posterior_2 = prior .* likelihood_norm_2;
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_2, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ 79b45389-fa2a-46df-9869-1992c8afb397
md"""

Now that we know more about the Binomial random variable, let us get back to our discussion on priors. 

"""

# ╔═╡ cb53475c-cc56-46b3-94b0-3ded33eb18d4
md"""
### Eliciting a prior 



"""

# ╔═╡ 75ef279d-2c7b-4776-b93f-5b28cbc67f63
md""" We start our discussion with the idea of prior elicitation. This is basically extracting a prior distribution from a person, normally an expert. Common practice is to settle on a distributional family and then elicit **hyperparameters** within the family.

The Beta distribution, for example, is a two parameter family with great flexibility. For different values of the parameters $a$ and $b$ we get different functional forms for the distribution. We refer to the parameters as hyperparemeters in Bayesian econometrics. One of the things that the researcher might have some information is the expected value of $\theta$.

"""

# ╔═╡ 2844b7a6-002e-4459-9e37-30e3a16c88f0
md" Here are some nice facts about the Beta distribution, you don't need to memorise these. 

$$\begin{equation}
\begin{array}{ll}
\text { Notation } & \operatorname{Beta}(a, b) \\
\hline \text { Parameters } & \begin{array}{l}
a>0 \text { shape (real) } \\
b>0 \text { shape (real) }
\end{array} \\
\hline \text { Support } & x \in[0,1] \text { or } x \in(0,1) \\
\text { PDF } & \frac{x^{a-1}(1-x)^{b-1}}{\mathrm{~B}(a, b)} \\
\hline \text { Mean } & \frac{a}{a+b} \\
\hline \text { Mode } & \frac{a-1}{a+b-2} \text { for } a, b>1 \\
& 0 \text { for } a=1, b>1 \\
& 1 \text { for } a>1, b=1 \\
\hline \text { Variance } & \frac{a b}{(a+b)^{2}(a+b+1)} \\
\text { Concentration } & \kappa=a+b
\end{array}
\end{equation}$$
"

# ╔═╡ 68bb2bfb-6643-4f59-9d2b-c59fd1dc3273
md"""
In the case of the Beta distribution the prior mean is given above as 

$\frac{a}{a + b}$

The prior mean will, by the fact that the prior is conjugate, also translate to a posterior distribution that has a Beta functional form. Therefore, if you choose the values for $a$ and $b$ properly you are in fact stating something about $\mathbb{E}(\theta)$.

Suppose you believe that $\mathbb{E}(\theta) = 1/2$. This can be obtained by setting $a = b$. 

As an example, set $a = b = 2$, then we have 

$\mathbb{E}(\theta) = \frac{2}{2+2} = 1/2$

We could also choose a completely noninformative prior with $a = b = 1$, which implies that $p(\theta) \propto 1$. This is simply a uniform distribution over the interval $[0, 1]$. Every value for $\theta$ receives the same probability. 

Obviously there are multiple values of $a$ and $b$ will work, play around with the sliders above to see what happens for a choice of different $a$ and $b$ under this restriction for the expected value of $\theta$.
In the case of the Beta distribution the prior mean is given above as 

$\frac{a}{a + b}$

The prior mean will, by the fact that the prior is conjugate, also translate to a posterior distribution that has a Beta functional form. Therefore, if you choose the values for $a$ and $b$ properly you are in fact stating something about $\mathbb{E}(\theta)$.

Suppose you believe that $\mathbb{E}(\theta) = 1/2$. This can be obtained by setting $a = b$. 

As an example, set $a = b = 2$, then we have 

$\mathbb{E}(\theta) = \frac{2}{2+2} = 1/2$

We could also choose a completely noninformative prior with $a = b = 1$, which implies that $p(\theta) \propto 1$. This is simply a uniform distribution over the interval $[0, 1]$. Every value for $\theta$ receives the same probability. 

Obviously there are multiple values of $a$ and $b$ will work, play around with the sliders above to see what happens for a choice of different $a$ and $b$ under this restriction for the expected value of $\theta$.

"""

# ╔═╡ 5c714d3b-ac72-40dc-ba98-bb2b24435d4c
md"""

#### Coin flipping contd. 

"""

# ╔═╡ 573b8a38-5a9b-4d5f-a9f6-00a5255914f0
md"""
In our coin flipping model we have derived the posterior credibilities of parameter values given certain priors. Generally, we need a mathematical description of the **prior probability** for each value of the parameter $\theta$ on interval $[0, 1]$. Any relevant probability density function would work, but there are two desiderata for mathematical tractability.

1. Product of $p(y \mid \theta)$ and $p(\theta)$ results in same form as $p(\theta)$.
2. Necesarry for $\int p(y \mid \theta)p(\theta) \text{d} \theta$ to be solvable analytically

When the forms of $p(y \mid \theta)$ and $p(\theta)$ combine so that the posterior has the same form as the prior distribution then $p(\theta)$ is called **conjugate prior** for $p(y \mid \theta)$. 

Prior is conjugate with respect to particular likelihood function. We are looking for a functional form for a prior density over $\theta$ that is conjugate to the **Bernoulli / Binomial likelihood function**.

If the prior is of the form, $\theta^{a}(1-\theta)^b$ then when you multiply with Bernoulli likelihood you will get

$$\begin{align*}
  \theta^{y + a}(1-\theta)^{(1-y+b)}
\end{align*}$$

A probability density of this form is called the Beta distribution. Beta distribution has two parameters, called $a$ and $b$.

$$\begin{align*}
  p(\theta \mid a, b) =& \text{Beta}(\theta \mid a, b) \\
  =& \frac{\theta^{a-1}(1-\theta)^{(b-1)}}{B(a,b)}
\end{align*}$$

In this case $B(a,b)$ is a normalising constant, to make sure area under Beta density integrates to $1$. 

Beta function is given by $\int_{0}^{1} \theta^{a-1}(1-\theta)^{(b-1)}\text{d}\theta$.

Another way in which we can express the Beta function,

$$\begin{align*}
  B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)  \quad \text{where} \quad \Gamma(a) = \int_{0}^{\infty} t^{(a-1)}\text{exp}(-t)
\end{align*}$$

The variables $a$ and $b$ are called the shape parameters of the Beta distribution (they determine the shape). We can use the `Distributions.jl` package to see what Beta looks like for different values of $a$ and $b$ over $\theta$.

"""

# ╔═╡ 1ca20976-757f-4e30-94d4-ee1276a614fb
md"""

!!! note "Parameters (a, b) of Beta distribution"
	Changing these sliders will illustrate how flexible the Beta distribution really is!

a = $(@bind α Slider(0.1:0.1:10, show_value = true, default=1)); 
b = $(@bind β Slider(0.1:0.1:10, show_value = true, default=1))
"""

# ╔═╡ aa69d0e8-cbbb-436c-b488-5bb113cdf97f
prior_beta = Beta(α, β);

# ╔═╡ dc43d1bc-ea5c-43ca-af0c-fc150756fa76
Plots.plot(
    Beta(α, β);
    xlabel=raw"$\theta$",
    ylabel=raw"$p_{\Theta}(\theta)$",
    title="\$\\mathrm{Beta}\\,($α, $β)\$",
    label=false,
    linewidth=2,
    fill=true,
    fillalpha=0.3,
	color = :black
)

# ╔═╡ 43d563ae-a435-417f-83c6-19b3b7d6e6ee
md"""

!!! note "Beta prior hyperparameters"
	Using different parameterisations of Beta will provide different posteriors.

a1 = $(@bind α₁ Slider(1:0.1:4, show_value = true, default=1));
b1 = $(@bind β₁ Slider(1:1:4, show_value = true, default=1))


"""

# ╔═╡ 11a5614b-c195-45a8-8be0-b99fda6c60fd
begin
	prior_beta₁ = Beta(α₁, β₁)
	prior_beta_pdf = pdf(prior_beta₁, coins_grid); # Beta distribution
	posterior_beta = prior_beta_pdf .* likelihood_norm_2;
	
	plot(prior_beta₁, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_beta, color = :black,lw = 2,  fill = (0, 0.4, :green), xlabel=raw"$\theta$",
    ylabel=raw"$p_{\Theta}(\theta)$",
    title="\$\\mathrm{Beta}\\,($α₁, $β₁)\$",)
end

# ╔═╡ f004ec01-1e27-4e30-9a53-23a299208846
md" Initially, with $a = 1$ and $b = 1$ this will be the same as the uniform prior. However, play around with the values on the slider to see how it changes for a different parameterisation of the Beta distribution. "

# ╔═╡ e7669fea-5aff-4522-81bf-3356ce126b1f
md"""

## Analytical derivation

"""

# ╔═╡ a32faf6c-a5bb-4005-ad42-188af732fba5
md"""
We have established the Beta distribution as a convenient prior for the Bernoulli and Binomial likelikood functions. Now we can figure out, mathematically, what the posterior would look like when we apply Bayes' rule. Suppose we have set of data with $N$ flips and $m$ heads, then we can calculate the posterior as,

$$\begin{align*}
  p(\theta \mid m, N) =& p(m, N \mid \theta)p(\theta)/p(m, N) \\
  =& \theta^{m}(1-\theta)^{(N-m)}\frac{\theta^{a-1}(1-\theta)^{(b-1)}}{B(a,b)} /p(m, N) \\
  =& \theta^{m}(1-\theta)^{(N-m)}{\theta^{a-1}(1-\theta)^{(b-1)}} / [B(a,b)p(m, N)] \\
  =& \theta^{((N + a) -1)}(1-\theta)^{((N-m + b)-1)}/ [B(a,b)p(m, N)] \\
  =& \theta^{((m + a) -1)}(1-\theta)^{((N-m + b)-1)}/ B(m + a, N-m+b)
\end{align*}$$

Last step was made by considering what the normalising factor should be for the numerator of the Beta distribution. From this we see that if prior is $\text{Beta}(\theta \mid a,b)$ then the posterior will be $\text{Beta}(\theta \mid m + a, N - m + b)$. Multiplying the likelihood and prior leads to a posterior with the same form as the prior. We refer to this as a **conjugate prior** (for a particular likelihood function). Beta priors are conjugate priors for the Bernoulli likelihood. If we use the Beta prior, we will in turn receive a Beta posterior. 

From this we can see that posterior is a compromise between the prior and likelihood. We illustrated this with graphs in the previous sections, but now we can see it analytically. For a $\text{Beta}(\theta \mid a, b)$ prior distribution, the prior mean of $\theta$ is $\frac{a}{a+b}$. If we observe $m$ heads in $N$ flips, this results in a proportion of $m/N$ heads in the data.  The posterior mean is 

$\frac{(m + a)}{m + a + N - m + b} = \frac{m+a}{N+a+b}$

This can then be rearranged algebraically into a weighted average of the prior mean and data proportion, 

$\underbrace{\frac{m+a}{N+a+b}}_{\text {posterior }}=\underbrace{\frac{m}{N}}_{\text {data }} \underbrace{\frac{N}{N+a+b}}_{\text {weight }}+\underbrace{\frac{a}{a+b}}_{\text {prior }} \underbrace{\frac{a+b}{N+a+b}}_{\text {weight }}$

This indicates that the posterior mean is somewhere between the prior mean and the proportion in the data. The more data we have, the less influence of the prior. 


"""

# ╔═╡ 92a4aa17-2e2d-45c2-a9a2-803d389077d5
md" ## Coin toss with `Turing.jl` 🤓"

# ╔═╡ 33b402b9-29c5-43d3-bb77-9b1a172229bb
md""" 

!!! note
	The material from this section comes from a talk by Jose Storopoli

"""

# ╔═╡ 0a3ed179-b60b-4740-ab73-b176bba08d84
md" In this section I will construct a coin tossing model in `Turing.jl`. We can approach the problem in one of two ways. We can see the process as independent Bernoulli trials or use a Binomial model. Don't worry too much about what MCMC methods are at this point, we will spend enough time going through these concepts later in the course. "

# ╔═╡ 47230bb3-de03-4353-9cbe-f974cc25411c
md""" #### How to specify a model """

# ╔═╡ 6f76e32c-32a7-4d77-b1f9-0078807ec103
md"""
**We specify the model inside a macro** `@model` where we can assign variables in two ways:

* using `~`: which means that a variable follows some probability distribution (Normal, Binomial etc.) and its value is random under that distribution

* using `=`: which means that a variable does not follow a probability distribution and its value is deterministic (like the normal `=` assignment in programming languages)

Turing will perform automatic inference on all variables that you specify using `~`.

Just like you would write in mathematical form:

$$\begin{aligned}
\theta &\sim \text{Beta}(1,1) \\
\text{coin flip} &\sim \text{Bernoulli}(\theta)
\end{aligned}$$

In our example we have an unfair coin with $\theta$ = 0.4 being the true value. 
"""

# ╔═╡ c205ff23-f1e7-459f-9339-2c80ab68945f
begin	
	# Set the true probability of heads in a coin.
	θ_true = 0.4	

	# Iterate from having seen 0 observations to 100 observations.
	Ns = 0:5

	# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
	Random.seed!(1237)
	data = rand(Bernoulli(θ_true), last(Ns))

	# Declare our Turing model.
	@model function coin_flip(y; α::Real=1, β::Real=1)
    	# Our prior belief about the probability of heads in a coin.
    	θ ~ Beta(α, β)

    	# The number of observations.
    	N = length(y)
    		for n ∈ 1:N
        	# Heads or tails of a coin are drawn from a Bernoulli distribution.
        	y[n] ~ Bernoulli(θ)
    	end
	end
end

# ╔═╡ 2bfe1d15-210d-43b4-ba4b-dec83f2363cd
md"""

In this example, `coin_flip` is a Julia function. It creates a model of the type `DynamicPPL.Model` which stores the name of the models, the generative function and the arguments of the models and their defaults. 

"""

# ╔═╡ b5a3a660-8928-4097-b1c4-90f045d17444
md"""
#### How to specify a MCMC sampler (`NUTS`, `HMC`, `MH` etc.)
"""

# ╔═╡ 6005e786-d4e8-4eef-8d6c-cc07fe36ea17
md"""
We have [several samplers](https://turing.ml/dev/docs/using-turing/sampler-viz) available:

* `MH()`: **M**etropolis-**H**astings
* `PG()`: **P**article **G**ibbs
* `SMC()`: **S**equential **M**onte **C**arlo
* `HMC()`: **H**amiltonian **M**onte **C**arlo
* `HMCDA()`: **H**amiltonian **M**onte **C**arlo with Nesterov's **D**ual **A**veraging
* `NUTS()`: **N**o-**U**-**T**urn **S**ampling

Just stick your desired `sampler` inside the function `sample(model, sampler, N; kwargs)`.

Play around if you want. Choose your `sampler`:
"""

# ╔═╡ 283fe6c9-6642-4bce-a639-696b92fcabb8
@bind chosen_sampler Radio([
		"MH()",
		"PG()",
		"SMC()",
		"HMC()",
		"HMCDA()",
		"NUTS()"], default = "MH()")

# ╔═╡ 0dc3b4d5-c66e-4fbe-a9fe-67f9212371cf
begin
	your_sampler = nothing
	if chosen_sampler == "MH()"
		your_sampler = MH()
	elseif chosen_sampler == "PG()"
		your_sampler = PG(2)
	elseif chosen_sampler == "SMC()"
		your_sampler = SMC()
	elseif chosen_sampler == "HMC()"
		your_sampler = HMC(0.05, 10)
	elseif chosen_sampler == "HMCDA()"
		your_sampler = HMCDA(10, 0.65, 0.3)
	elseif chosen_sampler == "NUTS()"
		your_sampler = NUTS(10, 0.65)
	end
end

# ╔═╡ 53872a09-db29-4195-801f-54dd5c7f8dc3
begin
	chain_coin = sample(coin_flip(data), your_sampler, 1000)
	summarystats(chain_coin)
end

# ╔═╡ b8053536-0e98-4a72-badd-58d9adbcf5ca
md"""
#### How to inspect chains and plot with `MCMCChains.jl`
"""

# ╔═╡ 97dd43c3-1072-4060-b750-c898ce926861
md"""
We can inspect and plot our model's chains and its underlying parameters with [`MCMCChains.jl`](https://turinglang.github.io/MCMCChains.jl/stable/)

**Inspecting Chains**
   * Summary Statistics: just do `summarystats(chain)`
   * Quantiles (Median, etc.): just do `quantile(chain)`

**Plotting Chains**: Now we have several options. The default `plot()` recipe will plot a `traceplot()` side-by-side with a `mixeddensity()`.

 First, we have to choose either to plot **parameters**(`:parameter`) or **chains**(`:chain`) with the keyword `colordim`.


Second, we have several plots to choose from:
* `traceplot()`: used for inspecting Markov chain **convergence**
* `meanplot()`: running average plots per interaction
* `density()`: **density** plots
* `histogram()`: **histogram** plots
* `mixeddensity()`: **mixed density** plots
* `autcorplot()`: **autocorrelation** plots


"""

# ╔═╡ 927bce0e-e018-4ecb-94e5-09812bf75936
plot(
	traceplot(chain_coin, title="traceplot"),
	meanplot(chain_coin, title="meanplot"),
	density(chain_coin, title="density"),
	histogram(chain_coin, title="histogram"),
	mixeddensity(chain_coin, title="mixeddensity"),
	autocorplot(chain_coin, title="autocorplot"),
	dpi=300, size=(840, 600), 
	alpha = 0.8
)

# ╔═╡ 398da783-5e47-4d39-8048-4541aad6b8b5
#StatsPlots.plot(chain_coin[:θ], lw = 1.75, color = :steelblue, alpha = 0.8, legend = false, dpi = 300)

# ╔═╡ a6ae40e6-ea8c-46f1-b430-961c1185c087
begin
#	StatsPlots.histogram(chain_coin[:θ], lw = 1.75, color = :black, alpha = 0.8, fill = (0, 0.4, :steelblue), legend = false)
end

# ╔═╡ 10beb8b2-0841-44c4-805e-8667da325b01
md""" 

#### Comparison with true posterior 

"""

# ╔═╡ 58f65290-8ba9-4c27-94de-b28a5eac80a4
md" We compare our result from using Turing with the analytical posterior that we derived in the previous section. "

# ╔═╡ c0daa659-f5f6-4e6b-9973-a399cf0ea788
begin
	# Our prior belief about the probability of heads in a coin toss.
	prior_belief = Beta(1, 1);
	
	# Compute the posterior distribution in closed-form.
	M = length(data)
	heads = sum(data)
	updated_belief = Beta(prior_belief.α + heads, prior_belief.β + M - heads)

	# Visualize a blue density plot of the approximate posterior distribution
	p = plot(chain_coin[:θ], seriestype = :density, xlim = (0,1), legend = :best, w = 2, c = :blue, label = "Approximate posterior")
	
	# Visualize a green density plot of posterior distribution in closed-form.
	plot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)), xlabel = "probability of heads", ylabel = "", title = "", xlim = (0,1), label = "Closed-form", fill=0, α=0.3, w=3, c = :green)
	
	# Visualize the true probability of heads in red.
	vline!(p, [θ_true], label = "True probability", c = :black, lw = 1.7, style = :dash)
	
end

# ╔═╡ Cell order:
# ╟─09a9d9f9-fa1a-4192-95cc-81314582488b
# ╟─41eb90d1-9262-42b1-9eb2-d7aa6583da17
# ╟─aa69729a-0b08-4299-a14c-c9eb2eb65d5c
# ╟─bcba487d-0b1f-4f08-9a20-768d50d67d7f
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═6f157a69-be96-427b-b844-9e76660c4cd6
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─4666fae5-b072-485f-bb3a-d28d54fb5274
# ╟─ec9e8512-85fb-4d23-a4ad-7c758cd87b62
# ╟─14339a8b-4486-4869-9cf2-6a5f1e4363ac
# ╟─2acbce7f-f528-4047-a2e0-0d7c710e37a1
# ╟─f7592ef8-e966-4113-b60e-559902911e65
# ╟─6d0bb8df-7211-4fca-b021-b1c35da7f7db
# ╟─4dfcf3a2-873b-42be-9f5e-0c93cf7220fc
# ╟─b8a11019-e87e-4ccb-b9dd-f369ba51a182
# ╠═d576514c-88a3-4292-93a8-8d23edefb2e1
# ╟─3abcd374-cb1b-4aba-bb3d-09e2819bc842
# ╟─411c06a3-c8f8-4d1d-a247-1f0054701021
# ╟─46780616-1282-4e6c-92ec-5a965f1fc701
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─7bbecc2b-8c49-458c-8f2e-8792efa62a32
# ╟─03288125-5ebd-47d3-9e1a-32e2308dbd51
# ╟─eed2582d-1ba9-48a1-90bc-a6a3bca139ba
# ╟─0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
# ╟─c6d22671-26e8-4ba3-865f-5cd449a6c9be
# ╟─34b792cf-65cf-447e-a30e-40ebca992427
# ╟─b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
# ╠═f364cb91-99c6-4b64-87d0-2cd6e8043142
# ╟─6087f997-bcb7-4482-a343-4c7830967e49
# ╟─bf5240e8-59fc-4b1f-8b0d-c65c196ab402
# ╠═8165a49f-bd0c-4ad6-8631-cae7425ca4a6
# ╟─5439c858-6ff3-4faa-9145-895390240d76
# ╟─0504929d-207c-4fb7-a8b9-14e21aa0f74b
# ╟─169fbcea-4e82-4756-9b4f-870bcb92cb93
# ╟─699dd91c-1141-4fb6-88fc-f7f05906d923
# ╟─6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
# ╟─284d0a23-a329-4ea7-a069-f387e21ba797
# ╟─bb535c41-48cb-44fd-989b-a6d3e310406f
# ╟─9e7a673e-7cb6-454d-9c57-b6b4f9181b06
# ╟─fe8f71b2-4198-4a12-a996-da254d2cc656
# ╟─7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
# ╟─f45eb380-7b43-4fd0-af34-89ffd126a63f
# ╟─4084f646-bce6-4a21-a529-49c7774f5ad1
# ╠═3c49d3e4-3555-4d18-9445-5347247cf639
# ╠═0520b5e3-cf92-4990-8a65-baf300b19631
# ╟─98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
# ╠═f7b158af-537e-4d9f-9c4c-318281097dce
# ╠═2cb41330-7ebd-45de-9aa1-632db6f9a140
# ╟─69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
# ╟─b6da2479-1545-4b1d-8d7f-07d6d1f67635
# ╟─c4cc482b-815b-4747-9f5a-5779d69086f7
# ╟─9016cba4-58f0-4b7f-91af-66faaf3fe99c
# ╟─828166f7-1a69-4952-9e3b-50a99a99789f
# ╟─24c4d724-5911-4534-a5c6-3ab86999df43
# ╠═5046166d-b6d8-4473-8823-5209aac59c84
# ╟─82d0539f-575c-4b98-8679-aefbd11f268e
# ╠═00cb5973-3047-4c57-9213-beae8f116113
# ╠═9e3c0e01-8eb6-4078-bc0f-019466afba5e
# ╟─c0bba3aa-d52c-4192-8eda-32d5d9f49a28
# ╟─ab9195d6-792d-4603-8605-228d479226c6
# ╟─e42c5e18-a647-4281-8a87-1b3c6c2abd33
# ╠═5d6a485d-90c4-4f76-a27e-497e8e12afd8
# ╠═987aeb82-267a-4ca9-bf21-c75a49edad70
# ╟─9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
# ╟─36e6f838-8277-480b-b48d-f70e8fe011eb
# ╟─11b8b262-32d2-4620-bc6b-afca4a5ce977
# ╟─89f7f633-4f75-4ef5-aa5b-80e318d14ee5
# ╟─11552b20-3407-4d0b-b07d-1488c8e8a759
# ╠═599c2f09-ad5e-4f39-aa7d-c1ba155725d6
# ╟─09ec10d9-a604-480d-8e82-59e84a843749
# ╠═071761f8-a187-47a6-8fee-5fc91e65d04c
# ╠═f001b040-2ae7-4e97-b229-eebaabb537b0
# ╠═9a2d5bdf-9597-40c7-ac18-bb27f187912d
# ╟─f6e6c4bf-9b2f-4047-a6cc-4ab9c3ae1420
# ╟─8382e073-433b-4e42-a6fa-d5a051586457
# ╠═9678396b-d42c-4c7c-821c-08126895efd3
# ╟─0a1d46ed-0295-4000-9e30-3ad838552a7e
# ╟─e5aade9a-4593-4903-bc3a-3a37f9f71c98
# ╠═87db6122-4d28-45bf-b5b0-41189792199d
# ╟─c6e9bb86-dc67-4f42-89da-98581a0c3c98
# ╟─b81924b8-73f6-4b28-899c-ec417d538dd4
# ╟─c3d2ba03-e676-4f0f-bafd-feecd0e4e414
# ╠═97c0c719-fb73-4571-9a6c-629a98cc544d
# ╠═0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
# ╟─4e790ffa-554d-4c46-af68-22ecb461fb7b
# ╟─4b141ffc-4100-47e3-941a-4e72c784ccf0
# ╟─219aafcb-17b1-4f5f-9c2b-9b713ba78b18
# ╟─2833e081-45d6-4f64-8d1e-b3a5895b7952
# ╟─79b45389-fa2a-46df-9869-1992c8afb397
# ╟─cb53475c-cc56-46b3-94b0-3ded33eb18d4
# ╟─75ef279d-2c7b-4776-b93f-5b28cbc67f63
# ╟─2844b7a6-002e-4459-9e37-30e3a16c88f0
# ╟─68bb2bfb-6643-4f59-9d2b-c59fd1dc3273
# ╟─5c714d3b-ac72-40dc-ba98-bb2b24435d4c
# ╟─573b8a38-5a9b-4d5f-a9f6-00a5255914f0
# ╟─1ca20976-757f-4e30-94d4-ee1276a614fb
# ╟─aa69d0e8-cbbb-436c-b488-5bb113cdf97f
# ╠═dc43d1bc-ea5c-43ca-af0c-fc150756fa76
# ╟─43d563ae-a435-417f-83c6-19b3b7d6e6ee
# ╟─11a5614b-c195-45a8-8be0-b99fda6c60fd
# ╟─f004ec01-1e27-4e30-9a53-23a299208846
# ╟─e7669fea-5aff-4522-81bf-3356ce126b1f
# ╟─a32faf6c-a5bb-4005-ad42-188af732fba5
# ╟─92a4aa17-2e2d-45c2-a9a2-803d389077d5
# ╟─33b402b9-29c5-43d3-bb77-9b1a172229bb
# ╟─0a3ed179-b60b-4740-ab73-b176bba08d84
# ╟─47230bb3-de03-4353-9cbe-f974cc25411c
# ╟─6f76e32c-32a7-4d77-b1f9-0078807ec103
# ╠═c205ff23-f1e7-459f-9339-2c80ab68945f
# ╟─2bfe1d15-210d-43b4-ba4b-dec83f2363cd
# ╟─b5a3a660-8928-4097-b1c4-90f045d17444
# ╟─6005e786-d4e8-4eef-8d6c-cc07fe36ea17
# ╟─283fe6c9-6642-4bce-a639-696b92fcabb8
# ╟─0dc3b4d5-c66e-4fbe-a9fe-67f9212371cf
# ╠═53872a09-db29-4195-801f-54dd5c7f8dc3
# ╟─b8053536-0e98-4a72-badd-58d9adbcf5ca
# ╟─97dd43c3-1072-4060-b750-c898ce926861
# ╟─927bce0e-e018-4ecb-94e5-09812bf75936
# ╠═398da783-5e47-4d39-8048-4541aad6b8b5
# ╠═a6ae40e6-ea8c-46f1-b430-961c1185c087
# ╟─10beb8b2-0841-44c4-805e-8667da325b01
# ╟─58f65290-8ba9-4c27-94de-b28a5eac80a4
# ╟─c0daa659-f5f6-4e6b-9973-a399cf0ea788
