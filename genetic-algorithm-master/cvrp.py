import osmnx as ox
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import folium
import networkx as nx

class folium_route(object):
  def __init__(self,G,start,end):
    self.H = G.copy()
    self.start = start
    self.end = end

  def shortest_route(self):
    start_ID = ox.distance.nearest_nodes(self.H, self.start[1], self.start[0])
    end_ID = ox.distance.nearest_nodes(self.H, self.end[1], self.end[0])
    shortest_distance = ox.shortest_path(self.H, start_ID, end_ID, weight = 'length')
    shortest_distance_length = nx.path_weight(self.H, shortest_distance, weight="length")
    
    return shortest_distance, shortest_distance_length

  def feature_route(self,m,fg,color='blue'):
    path, path_length = self.shortest_route()
    ox.folium.plot_route_folium(self.H, path, route_map=fg, zoom = 14,tiles='OpenStreetMap',color=color,tooltip='<b>Jarak Tempuh : {} m</b>'.format(round(path_length,2))).add_to(m)

# function for calculating distance between two pins
def _distance_calculator(G,_df):
    H = G.copy()
    _distance_result = np.zeros((len(_df),len(_df)))
    
    for i in range(len(_df)):
        for j in range(len(_df)):
            
            # calculate distance of all pairs
            start = (df.latitude.iloc[i], df.longitude.iloc[i])
            end = (df.latitude.iloc[j], df.longitude.iloc[j])
            route = folium_route(G,start,end)
            _, shortest_distance_length = route.shortest_route()

            # append distance to result list
            _distance_result[i][j] = shortest_distance_length
    
    return _distance_result

def draw_image(split_chromosome):
	"""
	visualization : plotting with matplolib
	"""
	plt.figure(figsize=(6,6))
	for i in range(customer_count):    
		if i == 0:
			plt.scatter(df.latitude[i], df.longitude[i], c='green', s=200)
			plt.text(df.latitude[i], df.longitude[i], "depot", fontsize=12)
		else:
			plt.scatter(df.latitude[i], df.longitude[i], c='orange', s=200)
			plt.text(df.latitude[i], df.longitude[i], str(df.demand[i]), fontsize=12)

	for vehicle in split_chromosome: 
		for i in range(len(vehicle)):
			startID = i
			endID = (i+1)%len(vehicle)
			plt.plot([df.latitude[vehicle[startID]], df.latitude[vehicle[endID]]],
					[df.longitude[vehicle[startID]], df.longitude[vehicle[endID]]], c="black")

	plt.show()

def partitions_num(n, L, U):
    if n<=U:
        yield [n]
    for i in range(L, n//2 + 1):
        for p in partitions_num(n-i, i, U):
            yield [i] + p

class Individual:
	def __init__(self, depot=0, chromosome=None):
		self.depot = depot
		self.score = np.infty
		self.chromosome = chromosome or self._makechromosome()
		self.split_chromosome = self.split_route_on_capacity_with_depot()

	def _makechromosome(self):
		"""
		Makes a chromosome randomly
		"""
		chromosome = [self.depot]
		lst = [i for i in range(1,customer_count)]
		for i in range(1,customer_count):
			choice = random.choice(lst)
			chromosome.append(choice)
			lst.remove(choice)

		return chromosome

	def _check(self):
		"""
		check the total demand that the driver carries
		"""
		if len(self.split_chromosome)>vehicle_count:
			return False

		for k, route in enumerate(self.split_chromosome):
			total = sum(df.demand.filter(route))

		if total > vehicle_capacity[k]:
			return False
		
		return True

	def evaluate(self):
		"""
		Calculate length of a route for current individual
		"""
		self.score = self.get_route_length() if self._check() else np.infty

	def crossover(self, other):
		"""
		cross two parents and returns created child's
		"""
		left, right = self._pickpivots()
		p1 = Individual()
		p2 = Individual()

		c1 = [c for c in self.chromosome[1:] if c not in other.chromosome[left:right+1]]
		p1.chromosome = [self.depot] + c1[:left] + other.chromosome[left:right+1] + c1[left:]
		c2 = [c for c in other.chromosome[1:] if c not in self.chromosome[left:right+1]]
		p1.chromosome = [other.depot] + c2[:left] + self.chromosome[left:right+1] + c2[left:]

		return p1, p2

	def mutate(self):
		"""
		swap two elements
		"""
		left, right = self._pickpivots()
		self.chromosome[left], self.chromosome[right] = self.chromosome[right], self.chromosome[left]

	def _pickpivots(self):
		"""
		return random left and right pivots
		"""
		left = random.randint(1,customer_count - 2)
		right = random.randint(left, customer_count -1)
		return left, right

	def copy(self):
		twin = self.__class__(self.chromosome[:])
		twin.score = self.score
		return twin

	def split_route_on_capacity_with_depot(self):
		permutation = random.choice(partitions)
		random.shuffle(permutation)

		step = 0
		split_routes = []
		for i, vehicle in enumerate(permutation):
			route = [self.chromosome[0]] + self.chromosome[1+step:permutation[i]+step+1]
			step += permutation[i]
			split_routes.append(route)

		return split_routes

	def get_route_length(self):
		"""
		return the total length of the route
		"""
		total = 0
		global distance

		for vehicle in self.split_chromosome:
			for i in range(len(vehicle)):
				j = (i+1) % len(vehicle)
				startID = vehicle[i]
				endID = vehicle[j]
				total += distance[startID][endID]

		return total

	def __repr__(self):
		return '<%s chromosome="%s" score=%s?' % (self.__class__.__name__, str(self.split_chromosome), self.score)

class Environment:
	def __init__(self, population=None, size=3, maxgenerations=5, newindividualrate=0.6, crossover_rate=0.8, mutation_rate=0.2):
		self.size = size
		self.population = population or self._makepopulation()
		self.maxgenerations = maxgenerations
		self.newindividualrate = newindividualrate
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.generation = 0
		self.minscore = np.infty
		self.minindividual = None
		self.list_score = []

	def _makepopulation(self):
		return [Individual() for _ in range(self.size)]

	def run(self):
		for i in range(1,self.maxgenerations + 1):
			print('Generation no:',str(i))
		for j in range(self.size):
			self.population[j].evaluate()
			curscore = self.population[j].score
			if curscore < self.minscore:
				self.minscore = curscore
				self.minindividual = self.population[j]

		print('Best individual:', self.minindividual.split_chromosome,'score:', round(self.minindividual.score,2),'\n')
		self.list_score.append(round(self.minindividual.score,2))

		if random.random() < self.crossover_rate:
			children = []
			newindividual = int(self.newindividualrate * self.size)
			for i in range(newindividual):
				selected1 = self._selectrank()
				while True:
					selected2 = self._selectrank()
					if selected1 != selected2:
						break

				parent1 = self.population[selected1]
				parent2 = self.population[selected2]
				child1, child2 = parent1.crossover(parent2)
				child1.evaluate()
				child2.evaluate()

				set_child1, set_child2 = False, False

				if child1.score < self.population[0].score:
					self.population.pop(0)
					self.population.append(child1)
					set_child1 = True

				if child2.score < self.population[1].score:
					self.population.pop(1)
					self.population.append(child2)
					set_child2 = True

				if not set_child1 and not set_child2:
					if child2.score < self.population[0].score:
						self.population.pop(0)
						self.population.append(child2)

					if child1.score < self.population[1].score:
						self.population.pop(1)
						self.population.append(child1)

			if random.random() < self.mutation_rate:
				selected = self._select()
				self.population[selected].mutate()

		for i in range(self.size):
			self.population[i].evaluate()
			curscore = self.population[i].score
			if curscore < self.minscore:
				self.minscore = curscore
				self.minindividual = self.population[i]

		print('--------------result-------------------')
		print(self.minindividual.split_chromosome, round(self.minindividual.score,2))

	def _select(self):
		totalscore = 0
		for i in range(self.size):
			totalscore += self.population[i].score

		randscore = random.random()*(self.size - 1)
		addscore = 0
		selected = 0
		for i in range(self.size):
			addscore += (1-self.population[i].score / totalscore)
			if addscore >= randscore:
				selected = i
				break

		return selected

	def _selectrank(self):
		return random.randint(0,self.size - 1)

def main():
	global df, customer_count, vehicle_count, vehicle_capacity, partitions, distance

	# Defining the map boundaries 
	north, east, south, west = -6.8686, 107.636, -6.9181, 107.5862

	# Downloading the map as a graph object 
	G = ox.graph_from_bbox(north, south, east, west, network_type = 'drive',simplify=True) 

	# customer count ('0' is depot) 
	customer_count = 10

	# the number of vehicle
	vehicle_count = 4

	# the capacity of vehicle
	vehicle_capacity = [50,45,40,60]

	# fix random seed
	np.random.seed(seed=777)

	# set depot latitude and longitude
	depot_latitude = -6.9073
	depot_longitude = 107.6181

	# make dataframe which contains vending machine location and demand
	df = pd.DataFrame({"latitude":np.random.normal(depot_latitude, 0.007, customer_count), 
					"longitude":np.random.normal(depot_longitude, 0.007, customer_count), 
					"demand":np.random.randint(10, 20, customer_count)})

	# set the depot as the center and make demand 0 ('0' = depot)
	df.latitude.iloc[0] = depot_latitude
	df.longitude.iloc[0] = depot_longitude
	df.demand.iloc[0] = 0

	distance = _distance_calculator(G, df)

	max_capacity = max(vehicle_capacity)
	min_capacity = min(vehicle_capacity)
	max_demand = max(df.demand)
	min_demand = min(df.demand.iloc[1:])
	max_client = math.floor(max_capacity/min_demand)
	min_client = math.ceil(min_capacity/max_capacity)
	partitions = list(partitions_num(customer_count-1,min_client,max_client))
	ev = Environment(size=1000, maxgenerations=100)
	ev.run()

if __name__ == '__main__':
	main()