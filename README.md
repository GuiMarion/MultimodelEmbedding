# MultimodelEmbedding
Multimodal embedding music for automatic piece recognition spaces. Machine Learning project for ATIAM master's degree at IRCAM.

## Idea
The idea is the following : we want to decompose a set of images into a larger set of squares of size k\*k pixels (15\*15 can be a good starting point) and sort all of this small images depending on their colors.

For that we propose the following method. 
  - We list all the files in a directory and we create the small pictures 
  - We use a discretized space of the pixels such as the points are equaly distributed. The figure below shows an exemple for a 2D space. Don't worry this function is already implemented. 
  
 ![prout](/src/discretized.png)

- We find the closest point for every picture and we save it in the appropriate folder. 


This idea is the classify the picture by color with a closest neigbour method in the descretized space. The problem is really easy and almost all has been done. 

## Functions

There is 4 functions to code.

  - computeDistance() -> 5min
  - getSubQuares() -> 10-20min
  - constructDataBase() -> 10-15min
  - sortImages() -> 10min
  
  As a lot of helper functions are already coded, this functions should be very easy to implement, don't forget to use the existing functions! 
  
## Exemples

Here are same example on how to use the existing functions : 
  
  i = img("Data/2005.Reykjavik.jpg") # open a file and construct an img object
  i.save("prout") # save an img object as a file named prout.jpg
  m = i.mean() # return the pixel mean of all the pixels : (rMean, vMean, bMean)

  S = discretizeSpace(3) # construct the discretized space with 3^3 points
  print(findClosestPoint(S, (100, 100, 10))) # return the closest point in S
  
## Dataset
  
  A small dataset is given in Data/ keep in mind that processing the algorithm on whole image will be longer then on a square. Note also that some function need another one to be executed.
