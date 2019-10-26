<div align='center'>
  
# Subtractive Clustering
</div>

My implementation of the subtractive clustering algorithm as described in [Sheng-Wu Xiong, Xiao-Xiao Niu and Hong-Bing Liu, (2005)](http://ieeexplore.ieee.org/abstract/document/1527702/). I implement the algorithm and test it on synthetic data where the underlying distribution is a set of three Gaussian clusters in 3-dimensional space. 

<p align="center">
<img src="https://github.com/alanjeffares/subtractive-clustering/blob/master/plots/compare_classifications.png"  width="400">
<img src="https://github.com/alanjeffares/subtractive-clustering/blob/master/plots/compare_centers.png"  width="400">

</p>

## File Descriptions
* `subtractive_clustering.py` - Implementation of the algorithm.
* `evaluation.py` - Evaluation of algorithm on synthetic `data.csv`.

## Usage
Navigate to the repository folder and run `python evaluation.py` on the command line.
