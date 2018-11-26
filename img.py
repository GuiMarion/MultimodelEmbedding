import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class img:
	def __init__(self, path, matrix = False):
		'''
		Open an image and store the matrix
		'''
		if matrix:
			self.matrix = path
		else:
			self.matrix = np.array(Image.open(path))

	def mean(self):
		'''
		Return the mean pixel of all the pixels of the matrix
		'''
		(r, v, b) = (0, 0, 0)
		for x in range(len(self.matrix)):
			for y in range(len(self.matrix[0])):
				r += self.matrix[x][y][0]
				v += self.matrix[x][y][1]
				b += self.matrix[x][y][2]

		r /= len(self.matrix)*len(self.matrix[0])
		v /= len(self.matrix)*len(self.matrix[0])
		b /= len(self.matrix)*len(self.matrix[0])

		return (r, v, b)

	def getMatrix(self):
		return self.matrix

	def print(self):
		for x in range(len(self.matrix)):
			print()
			for y in range(len(self.matrix[0])):
				print(self.matrix[x][y], end =" ")
		print()

	def save(self, name):
		im = Image.fromarray(self.matrix)
		im.save(name + ".jpg")

	def computeDistance(self, val):
		'''
		Should return the distance between val et the mean of the image
		'''
		(r, v, b) = val

		m1, m2, m3 = self.mean()

		return np.sqrt((r-m1)**2 +  (v-m2)**2 + (b-m2)**2)

	def getSubSquares(self, n):
		'''
		Should return subsquared images of size n*n as a liste of img objects
		'''

		images = []

		for i in range(len(self.matrix) // n - 1):
			for j in range(len(self.matrix[0]) // n - 1):
				for k1 in range(0, n, 25):
					for k2 in range(0, n, 25):
						images.append(img(self.matrix[i*n + k1 : (i+1)*n + k1, j*n + k2 : (j+1)*n + k2], matrix = True))

		return images

	def findClosestPoint(self, space):
		dist =  100000
		for elem in space:
			distTemp = self.computeDistance(elem)
			if distTemp <= dist:
				dist = distTemp
				returnedValue = elem

		return returnedValue

def constructDataBase(path, n):
	'''
	Open all the files from the path and return a list of all the resulting subsquares
	'''

	files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

	images = []

	for file in files:
		images.extend(img(file).getSubSquares(n))

	return images

def sortImages(images, OUT, spaceSize = 5):
	'''
	For each image in images, find the closest point of the dicretized space and save the image in the directory with the name of this point
	'''
	space = discretizeSpace(spaceSize)

	for point in space : 
		if not os.path.exists(OUT + "/" + str(point)):
		    os.makedirs(OUT + "/" + str(point))

	k = 0
	for image in tqdm(images):

		point = image.findClosestPoint(space)

		image.save(OUT + "/" + str(point) + "/" + str(k))
		k += 1


def discretizeSpace(n):
	'''
	Return n^3 points equaly distributed in the pixel space (0, 0, 0) -> (255, 255, 255)
	'''

	points = []

	for r in range(n):
		for v in range(n):
			for b in range(n):
				points.append((r*255//n + 255//(2*n), v*255//n + 255//(2*n), b*255//n + 255//(2*n)))

	return points



def main():
	'''
	Do the job if all the functions are working correctly
	'''
	n = 128
	DATA = "Data/"
	OUT = "OUT/"+DATA
	spaceSize = 5
	data = constructDataBase(DATA, n)
	sortImages(data, OUT, spaceSize = 5)

main()
