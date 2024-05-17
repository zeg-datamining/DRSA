# DRSA
We have designed a sparse attention mechanism with dynamic routing perception called Dynamic Routing Sparse Attention (DRSA) to address these issues. 
Specifically, DRSA can effectively handle variations of complex time series data. Meanwhile, under memory constraints, the Dynamic Routing Filter (DRF) module further refines it by 
filtering the blocked 2D time series data to identify the most relevant feature vectors in the local context. We conducted predictive experiments on six real-world time series datasets 
with fine granularity and long sequence dependencies. 

* We have designed the DRSA model, which approaches time series data analysis from global and local perspectives to tackle time series representation learning tasks.
* Leveraging a flexible block-based approach, we shift the research focus from the global to the local view, unearthing more hidden features. Building on this foundation,
  we have devised a Dynamic Routing Filtering (DRF) mechanism, which dynamically adapts to the local context through route selection.

## Requirments
This code requires the following:

- Python>=3.8
- PyTorch>=1.12.1
- Numpy>=1.26.0
- Scikit-learn>=1.3.2

## Get Started
1.  Install Python 3.8, PyTorch >= 1.12.1. 
2.  Download data.
3.  Train and evaluate. We provide the experiment scripts of all DRSA under the folder ./scripts.

## Contact
If you have any question, please contact wangwenyan32@163.com.


