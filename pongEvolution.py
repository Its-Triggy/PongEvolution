import pygame
import numpy as np
import random
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,180,0)
BLUE = (50,200,255)
FILL = BLACK
TEXT = WHITE

pygame.init()

#Here you can specify the structure of the neural network. This includes the input layer and output layer.
#e.g 3 inputs, 5 node hidden layer, 4 outputs would be [3, 5, 4]
#Be sure to update this if you add inputs
layer_structure = [4, 3]

#Initializing the display window
size = (800,600)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("pong")

testCoefs = [np.array([[0.38238344, 0.7515745 , 0.29565119, 0.35490288, 0.97040034],
       [0.33545982, 0.0973694 , 0.41539856, 0.76129553, 0.93089118],
       [0.85154809, 0.0240888 , 0.74555908, 0.34759429, 0.37355357],
       [0.95104127, 0.29077331, 0.21244898, 0.78876218, 0.35243364]]), np.array([[0.25498077, 0.03853811, 0.76089995],
       [0.36535132, 0.60519588, 0.08365677],
       [0.12852428, 0.0156597 , 0.03317768],
       [0.1276382 , 0.13700435, 0.6786845 ],
       [0.71931642, 0.8930938 , 0.24983195]])]

#Paddle class
class Paddle:
	
	def __init__(self, x = 400, xspeed = 0, coefs = 0, intercepts = 0):
		self.x = x
		self.xlast = x-xspeed
		self.xspeed = xspeed
		self.alive = True
		self.score = 0
		self.command = 2
		self.winner = False
		if coefs == 0:
			self.coefs = self.generateCoefs(layer_structure)
		else:
			self.coefs = coefs
		if intercepts == 0:
			self.intercepts = self.generateIntercepts(layer_structure)
		else:
			self.intercepts = intercepts
	 
	#Creates random coefficients for the neural network 
	def generateCoefs(self, layer_structure):
		coefs = []
		for i in range(len(layer_structure)-1):
			coefs.append(np.random.rand(layer_structure[i], layer_structure[i+1])*2-1)
		return coefs
	
	#Creates random intercepts for the neural network 	
	def generateIntercepts(self, layer_structure):
		intercepts = []
		for i in range(len(layer_structure)-1):
			intercepts.append(np.random.rand(layer_structure[i+1])*2-1)
		return intercepts
	
	#Returns mutated coefs
	def mutateCoefs(self):
		newCoefs = self.coefs.copy()
		for i in range(len(newCoefs)):
			for row in range(len(newCoefs[i])):
				for col in range(len(newCoefs[i][row])):
					newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
		return newCoefs
	
	#Returns mutated intercepts	
	def mutateIntercepts(self):
		newIntercepts = self.intercepts.copy()
		for i in range(len(newIntercepts)):
			for row in range(len(newIntercepts[i])):
				newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
		return newIntercepts
	
	#Returns a paddle with mutated coefs and intercepts
	def mutate(self):
		return Paddle(coefs = self.mutateCoefs(), intercepts = self.mutateIntercepts())
		
	#Reset score, speed and position
	def reset(self):
		self.x = 400
		self.xlast = 400
		self.xspeed = 0
		self.alive = True
		self.score = 0
	
	#Update position based on speed		
	def update(self):
		self.xlast = self.x
		self.x += self.xspeed
		if self.x < 0:
			self.x = 0
		elif self.x > size[0]-100:
			self.x=size[0]-100
		
		self.xlast = self.x
	
	#Draw the paddle to the screen	   
	def draw(self):
		if self.winner == False:
			pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
			pygame.draw.rect(screen,RED,[self.x+2,size[1]-18,100-4,20-4])
		else:
			pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
			pygame.draw.rect(screen,BLUE,[self.x+2,size[1]-18,100-4,20-4])

#Ball class
class Ball:
	
	def __init__(self, x = 50, y = 50, xspeed = 5, yspeed = 5):
		self.x = x
		self.y = y
		self.xlast = x-xspeed
		self.ylast = y-yspeed
		self.xspeed = xspeed
		self.yspeed = yspeed
		self.alive = True
	
	#Update position based on speed 
	def update(self, paddle):
		self.xlast = self.x
		self.ylast = self.y
		
		self.x += self.xspeed
		self.y += self.yspeed
		
		#Accounts for bouncing off walls and paddle
		if self.x<0:
			self.x=0
			self.xspeed = self.xspeed * -1
		elif self.x>size[0]-15:
			self.x=size[0]-15
			self.xspeed = self.xspeed * -1
		elif self.y<0:
			self.y=0
			self.yspeed = self.yspeed * -1
		elif self.x>paddle.x and self.x<paddle.x+100 and self.ylast<size[1]-35 and self.y>=size[1]-35:
			self.yspeed = self.yspeed * -1
			paddle.score = paddle.score + 1
		elif self.y>size[1]:
			self.yspeed = self.yspeed * -1
			paddle.alive = False
			paddle.score -= round(abs((paddle.x+50)-self.x)/100,2)
			
	#Draw ball to screen	   
	def draw(self):
		pygame.draw.rect(screen,WHITE,[self.x,self.y,15,15])
	
#Predicts the output for a given input given an array of coefficients and an array of intercepts
def calculateOutput(input, layer_structure, coefs, intercepts, g="identity"):
	#The values of the neurons for each layer will be stores in "layers", so here the input layer is added to start
	#(Stuff is transposed since we need columns for matrix multiplication)
	layers = [np.transpose(input)]
	#The current layer will be affected by the previous layer, so here we define the starting previousLayer as the input 
	previousLayer = np.transpose(input)
	
	reduced_layer_structure = layer_structure[1:]
	#Loops through the all the layers except the input
	for k in range(len(reduced_layer_structure)):
		#creates an empty array of the correct size
		currentLayer = np.empty((reduced_layer_structure[k],1))
		#The resulting layer is a matrix multiplication of the previousLayer and the coefficients, plus the intercepts
		result = np.matmul(np.transpose(coefs[k]),previousLayer) + np.transpose(np.array([intercepts[k]]))
		#The value of each neuron is then put through a function g()
		for i in range(len(currentLayer)):
			if g == "identity":
				currentLayer[i] = result[i]
			elif g == "relu":
				currentLayer[i] = max(0, result[i])
			elif g == "tanh":
				currentLayer[i] = tanh(result[i])
			elif g == "logistic":
				try:
					currentLayer[i] = 1 / (1 + exp(-1*result[i]))
				except OverflowError:
					currentLayer[i] = 0
		#The current layer is then added to the layers list, and the previousLayer variable is updated
		layers.append(currentLayer)
		previousLayer = currentLayer.copy()
	
	#Returns the index of the highest value neuron in the output layer (aka layers[-1])
	#E.g. if the 7th neuron has the highest value, returns 7
	return(layers[-1].tolist().index(max(layers[-1].tolist())))	
	
#Returns a set of coefficients which are a mutation of the input
def mutateCoefs(coefs):
	newCoefs = []
	for array in coefs:
		newCoefs.append(np.copy(array))
	for i in range(len(newCoefs)):
		for row in range(len(newCoefs[i])):
			for col in range(len(newCoefs[i][row])):
				newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
	return newCoefs

#Returns a set of intercepts which are a mutation of the input
def mutateIntercepts(intercepts):
	newIntercepts = []
	for array in intercepts:
		newIntercepts.append(np.copy(array))
	for i in range(len(newIntercepts)):
		for row in range(len(newIntercepts[i])):
			newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
	return newIntercepts 
	
#Displays the nodes of a network, along with weighted lines showing the coefficient influences
def displayNetwork(layer_sctructure, coefs = testCoefs, command = 0):
	
	#Stores the larges coefficient, so we can scale the thicknesses accordingly. 
	max_coef = np.max(coefs[0])
	
	#Determines how much space this visual network will take up
	height = 300
	width = 300
	
	inputs = ["paddle x", "ball x", "ball y", "ball Xspeed", "ball Yspeed"]
	outputs = ["left", "right", "stop"]
	
	layerCount = len(layer_structure)
	#This will store the positions of all the nodes (organized with sub-lists of each layer)
	circle_positions = []
	
	#Label inputs
	for i in range(layer_structure[0]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(inputs[i], True, TEXT)
		screen.blit(text,[0,(i+1)* int(height/(layer_structure[0]+2))])	
	
	#Label outputs
	for i in range(layer_structure[-1]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(str(outputs[i]), True, TEXT)
		screen.blit(text,[width+50,(i+1)* int(height/(layer_structure[-1]+2))])	
	
	#Calculates an appropriate spacing of the layers
	xspacing = int( width/(layerCount))
	
	#Determine the location of the neurons for each layer, stores that in a list, and stores those lists in circle_positions
	for i in range(layerCount):
		layer_circle_positions = []
		yspacing = int( height/(layer_structure[i]+2))	
		for j in range(layer_structure[i]):
			layer_circle_positions.append(((i+1)*xspacing, (j+1)*yspacing))
		circle_positions.append(layer_circle_positions)
	
	#Draws a line between every node in one layer and every node in the next layer
	for i in range(len(circle_positions)-1):
		for j, circle_pos in enumerate(circle_positions[i]):
			for k, circle_pos2 in enumerate(circle_positions[i+1]):
				thickness = int(coefs[i][j,k]/max_coef*8)
				
				if thickness > 0:
					pygame.draw.lines(screen, BLUE, False, [circle_pos, circle_pos2], thickness)
				else:
					pygame.draw.lines(screen, RED, False, [circle_pos, circle_pos2], -thickness)
					

	#Draws circles in the positions of the nodes (over the lines)
	for layer in circle_positions:
		for circle_pos in layer:
			pygame.draw.circle(screen, BLACK, circle_pos, 20, 0)
			pygame.draw.circle(screen, GREEN, circle_pos, 16, 0)
			

done = False
score = 0
command = "stop"
clock=pygame.time.Clock()

COUNT = 100

#create sprites
paddles = []
balls = []
for i in range(100):
	paddles.append(Paddle())
	balls.append(Ball())

#The first winner is arbitrarily chosen to be the last one (just so the user has a network to watch on screen)
winner = paddles[-1]
paddles[-1].winner = True

#game's main loop  
generation = 1
while not done:
	screen.fill(FILL)
	
	#Track the number of paddles still alive in this generation
	still_alive = 0
	#Track the high score and the index of the highest scoring paddle
	high_score = -9e99
	high_score_index = -1
	
	#Allow user to exit at any time
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			done = True
	
	#Loop through all the paddles
	for i, paddle in enumerate(paddles):
		#If you change the number of inputs, be sure to change the layer_structure at the top and the input text in displayNetwork
		input = np.array([[paddle.x, balls[i].x, balls[i].y, balls[i].xspeed]])
		paddle.command = calculateOutput(input, layer_structure, paddle.coefs, paddle.intercepts)
		
		#0=left, 1=right, 2=stop
		if paddle.command == 0:
				paddle.xspeed = -5
		elif paddle.command == 1:
				paddle.xspeed = 5
		elif paddle.command == 2:
				paddle.xspeed = 0
		
		#Update position of all living paddles
		if paddle.alive == True:
			paddle.update()  
			balls[i].update(paddle)
			still_alive += 1

		#Update high_score and high_scorer
		if paddle.score > high_score:
			high_score = paddle.score
			high_score_index = i
			winner = paddles[i]
			winner.winner = True
			
		#Draw everything but the winner
		if paddle.alive and paddle != winner:
			paddle.draw()
			balls[i].draw()
			paddle.winner = False
	
	#draw the winner last (so that it is not hidden behind other paddles)
	paddles[high_score_index].draw()
	balls[high_score_index].draw()
		
	#If all the paddles are dead, reproduce the most fit one
	if still_alive == 0:
		generation += 1
		winner.reset()
		print(high_score_index)
		#clear the generation
		paddles = []
		balls = []
		#Fill it with mutations of the winner
		for i in range(COUNT-1):
			paddles.append(Paddle(coefs = mutateCoefs(winner.coefs), intercepts = mutateIntercepts(winner.intercepts)))
			balls.append(Ball())
		#add the winner itself
		paddles.append(winner)
		balls.append(Ball())

	
	#score board
	font= pygame.font.SysFont('Calibri', 50, False, False)
	text = font.render("Score = " + str(high_score), True, TEXT)
	screen.blit(text,[size[0]-300,30])	
	text2 = font.render("Still alive = " + str(still_alive), True, TEXT)
	screen.blit(text2,[size[0]-300, 90])	
	text2 = font.render("Generation = " + str(generation), True, TEXT)
	screen.blit(text2,[size[0]-300, 150])	
	displayNetwork(layer_structure, coefs = winner.coefs)
  
	pygame.display.flip()		 
	clock.tick(60)
	
pygame.quit()	
