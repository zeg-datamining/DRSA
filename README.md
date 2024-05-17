# DRSA
We have designed a sparse attention mechanism with dynamic routing perception called Dynamic Routing Sparse Attention (DRSA) to address these issues. 
Specifically, DRSA can effectively handle variations of complex time series data. Meanwhile, under memory constraints, the Dynamic Routing Filter (DRF) module further refines it by 
filtering the blocked 2D time series data to identify the most relevant feature vectors in the local context. We conducted predictive experiments on six real-world time series datasets 
with fine granularity and long sequence dependencies. 

* We have designed the DRSA model, which approaches time series data analysis from global and local perspectives to tackle time series representation learning tasks.
* Leveraging a flexible block-based approach, we shift the research focus from the global to the local view, unearthing more hidden features. Building on this foundation,
  we have devised a Dynamic Routing Filtering (DRF) mechanism, which dynamically adapts to the local context through route selection.
