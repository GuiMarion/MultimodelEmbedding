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

  - computeDistance() -> 5min,
    compute distance between a 3D value and the mean pixel of the matrix, it's juste a 3D Euclidean distance.
    
  - getSubSquares() -> 10-20min,
    construct small images from a big one by croping squares from the original matrix. 
    
  - constructDataBase() -> 10-15min,
    list files from a directory and apply the getSubSquares function.
  
  - sortImages() -> 10-20min,
    from a set of images, find the closest point for each image, then save the image into the good folder.
  
  As a lot of helper functions are already coded, this functions should be very easy to implement, don't forget to use the existing functions! If there somethin you don't know how to do (listing files from a directory for exemple), don't hesitate to take a look at google (stackoverflow if a very good website!). 
  
  All the functions you need to code are pre-declared and there is a #### TODO #### where you should put the code.
  
  Use numpy to manipulate matrix!
  
## Exemples

Here are same examples of how to use the existing functions : 
  
  ```python
  i = img("Data/2005.Reykjavik.jpg") # open a file and construct an img object
  
  i.save("prout") # save an img object as a file named prout.jpg
  
  m = i.mean() # return the pixel mean of all the pixels : (rMean, vMean, bMean)


  S = discretizeSpace(3) # construct the discretized space with 3^3 points
  
  print(findClosestPoint(S, (100, 100, 10))) # return the closest point in S
  
  data = np.zeros((100, 100, 3), dtype=np.uint8) # Construct a matrix with only 0
  
  i = img(data, matrix = True) # Create a img object form the matrix
  
  i.save("ok") # Save the image
  ```
  
## Dataset
  
  A small dataset is given in Data/ keep in mind that processing an algorithm on a whole image will be longer then on a little squared one. Note also that some function need another one to be executed.
  
## Oriented Object Programmation

  In OOP we define objects that contain attributes (data) and class methods (functions) with the key-word *class*, each class contains a consctructor (__init__ in python) that is used to initialize the object. In python, 
  
  ```python
    class car:
      def __init__(self, color, model):
        self.color = color
        self.model = model
  ```
        
The class *car*, contains two attributes (color and model) that are initialized in the constructor by the parameters of the function. Each function of the class must contain self as first parameter. In a function, self is used to access an attribute or method of the object, it can be a function or a data. For exemple, 
```python
    class car:
      def __init__(self, color, model, key):
        self.color = color
        self.model = model
        self.key = key
        self.started = False
        self.position = (0,0)
        
      def start(self, mykey):
        if mykey == self.key:
          self.started = True
        
      def drive(self, direction, mykey):
        if self.started == False:
          self.start(mykey)
        self.position += direction
```      
 Here I used self.key in order to access the attribute named *key* I initialized in the constructor, I also used self.start(mykey) to call the class function *start* I defined into the class, note that I didn't put self as a argument (not self.start(self, mykey)).
 
 Now you know how to define a class, we will see how to instintiate a class, an object is a part of the memory used for a class that we instantiated. For this, nothing simpler, we use the conscructor method that can be called this way:
```python 
    mycar = car("red", "peugeot105")
 ```
 
 We use the name of the class to call the constructor, we now have an object of type *car* in our variable mycar. We can now access attributes and methods in a similar way of when we was inside the class, instead of using self we use the name of the variable. 
 ```python
    mycar.drive("3435", (10, 10))
    print(mycar.position)
 ```
    
 Now you should know everything about Oriented Object Programming!
 
