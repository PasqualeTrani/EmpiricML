<p align="center">
  <img src="EmpiricML-logo.png" width="250" height="250" alt="EmpiricML Logo">
</p>

# Home

## What is EmpiricML?
EmpiricML is a Python framework designed to make building, testing, and tracking machine learning models on **tabular data** faster, easier, and most importantly, **robust**. Built on the shoulders of giants - **scikit-learn** and **Polars** - it brings the concept of a scientific "Laboratory" to your ML workflow.

## The Core Ideas Behind the Framework

### Machine Learning as an Empirical Science
The first core idea behind the framework is that **Machine Learning is an empirical science**.
In empirical sciences, we design **experiments** to test hypotheses. To do that effectively, we need a controlled environment, i.e. a **Laboratory**.

**EmpiricML provides that Laboratory.**

Instead of scattered scripts and notebooks, the `Lab` class encapsulates everything required for a rigorous ML experiment:

*   **Data**: Training and testing datasets (handled efficiently via Polars LazyFrames).
*   **Protocol**: A defined Cross-Validation strategy.
*   **Measurement**: A specific Error or Performance Metric.
*   **Criteria**: Rules for statistical comparison to determine if Model A is *truly* better than Model B.

### Justifying Complexity with Evidence
Another key idea is that **complexity must be earned**. We should start with simple baselines and only adopt more complex models if they demonstrate a clear, statistically significant improvement.

In industrial environments, complexity carries a cost in infrastructure, interpretability, latency, and maintenance. EmpiricML encourages a workflow where we gather **evidence** that a complex model is strictly better than a simpler one before accepting that cost.

## Advantages

### Shrink Time to Production
EmpiricML allows you to significantly shrink the time it takes to put a model into production. By enforcing structured experimentation and managing artifacts automatically, the transition from a research prototype to a production-ready model is seamless, eliminating the need for extensive code rewriting.

### Better Data Processing with Polars
EmpiricML encourages and facilitates the use of **Polars** over Pandas. Polars offers numerous advantages:

*   **Performance**: Lightning-fast execution due to its Rust implementation and Arrow memory format.
*   **Memory Efficiency**: Handles larger-than-RAM datasets with LazyFrames.
*   **Expressiveness**: A modern, readable query API that reduces bugs and improves maintainability.

For more information, see the [Polars documentation](https://pola-rs.github.io/polars-book/).