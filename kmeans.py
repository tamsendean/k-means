import matplotlib.pyplot as plt
import numpy as np
import math

file = "JoensuuRegion.txt" 	# change input file

# init min and max
x1_min = float('inf')
x1_max = float('-inf')
x2_min = float('inf')
x2_max = float('-inf')

# read in file and preprocess
def read_in():
	global x1_min
	global x1_max
	global x2_min
	global x2_max

	X1 = []
	X2 = []
	pt = 0

	for line in open(file, "r"):
		val = line.strip()
		val = [float(x) for x in val.split(',')]

		# calculate min and max values of each feature
		if val[0] < x1_min:
			x1_min = val[0]
		if val[0] > x1_max:
			x1_max = val[0]

		if val[1] < x2_min:
			x2_min = val[1]
		if val[1] > x2_min:
			x2_max = val[1]

		X1.append(val[0])
		X2.append(val[1])
		pt += 1

	return X1, X2, pt


# prompts user for k-value
def get_k():
	k = 0
	while k <= 0:
		k = int(input("Please enter the number of clusters : "))
	return k


# find euclidean distance between points (x1, y1) and (x2, y2)
def calc_distance(x1, y1, x2, y2):
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# calculates cluster entropy
# code cited from @jambao24 on GitHub, Mar 15, 2021:
# https://github.com/jambao24/ML_Assignments/blob/a6d9808fa32956f8f6aefae0551d244ad2bd65b5/ass4/optdigits_kmeans.py#L67
def calc_entropy(targets_list):

    targets = np.array(targets_list)
    types, counts = np.unique(targets, return_counts=True)
    count_sum = np.sum(counts)

    probs = np.divide(counts, count_sum)
    log2probs = np.log2(probs)
    to_sum = np.multiply(probs, log2probs)
    return -1*np.sum(to_sum)

# runs k-means algorithm
def run_k(X1, X2, pt, k):

	print("Running k-means algorithm...")

	C = [None] * pt

	iter = 0

	for x in range(1000):

		# init cluster centers at random
		C1 = (x1_max - x1_min) * np.random.random_sample(k) + x1_min
		C2 = (x2_max - x2_min) * np.random.random_sample(k) + x2_min

		for i in range(pt):

			min_dist = np.inf
			min_c = 0
			for j in range(k):
				# calculate euclidean distance between each data point and each cluster center
				e = calc_distance(X1[i], X2[i], C1[j], C2[j])

				# keep track of the closest cluster center for each point
				if e < min_dist:
					min_dist = e
					min_c = j
			C[i] = min_c

		cluster_pts = [0] * k
		C1_new = [0] * k
		C2_new = [0] * k

		# find number of points in each cluster and sum of individual feature values
		for i in range(pt):
			cluster_pts[C[i]] += 1

			C1_new[C[i]] += X1[i]
			C2_new[C[i]] += X2[i]

		# calculate and assign new cluster centers based on average of the features of points in that cluster
		C1 = [C1_new[i]/cluster_pts[i] for i in range(k)]
		C2 = [C2_new[i]/cluster_pts[i] for i in range(k)]

		iter += 1

	print("Final set of centres is : ", str(["("+str(C1[i])+","+str(C2[i])+")" for i in range(k)]))
	print("Mean clustering entropy: ", calc_entropy(C))

	return C, C1, C2


# scatterplot of points and cluster centers
def display_final_plot(X1, X2, pt, C, C1, C2, k):
	for i in range(pt):
		plt.scatter(X1[i], X2[i], color='black')
	plt.xlabel("x1")
	plt.ylabel("x2")
	plt.title("Clusters")

	# display the cluster centers in red
	for i in range(k):
		plt.scatter(C1[i], C2[i], c='red', s=100, label="centroid", alpha = 1)
	plt.show()


# main function
def main():
	X1, X2, pt = read_in()
	
	k = get_k()
	
	C, C1, C2 = run_k(X1, X2, pt, k)

	display_final_plot(X1, X2, pt, C, C1, C2, k)


if __name__ == '__main__':
	main()