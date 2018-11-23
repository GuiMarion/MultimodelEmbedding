# MultimodelEmbedding
Multimodal embedding music for automatic piece recognition spaces. Machine Learning project for ATIAM master's degree at IRCAM.


Another exercis, the idea is the following : we want to decompose a set of images into a larger set of squares of size k*k pixels (15*15 can be a good starting point) and sort all of this small images depending on their colors.

For that we propose the following method. 
  - We list all the files in a directory and we create the small pictures 
  - We use a discretized space of the pixels such as the points are equaly distributed. The figure below shows an exemple for a 2D space. 
  
 ![prout](/src/discretized.png)
