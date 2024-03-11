# CS3308-project
## Implement PCA and t-SNE
- PCA finds the most important principal components in the data by calculating the covariance matrix. It represents the data using these principal components, making it more effective for linear relationships. This can be observed from the visualization results.

- On the other hand, t-SNE maps high-dimensional data to a lower-dimensional space by optimizing the similarity between data points. It preserves complex structures and clustering relationships in the high-dimensional data, resulting in better clustering effects. This can also be seen from the visualization results.

- Furthermore, during the visualization process, both LeNet and MyNet perform poorly in clustering the pink category. Attempts have been made to optimize the visualization by modifying parameters such as the number of iterations and perplexity in t-SNE, but the results have been minimal. It is speculated that the complexity of the data features or excessive noise may lead to inaccurate similarity calculations, thereby affecting the dimensionality reduction effect of t-SNE.

## Image reconstruction
The following is the structure of my VAE net:
![image](https://github.com/Double-City/CS3308-project/assets/95283869/dfbbc8c9-f2f3-4b1a-9c72-d867e069dac9)
