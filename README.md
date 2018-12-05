# MultimodalEmbedding
Multimodal embedding music for automatic piece recognition spaces. Machine Learning project for ATIAM master's degree at IRCAM.

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
 
