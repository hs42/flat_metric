Here you can find some mock data files.
Two distributions were generated, each one with a dataset for training, testing, and validation. In practice, you would chip off some of your available data
and keep it solely for testing or do some cross validation. The data itself usually comes from lab measurements etc.

The samples found here were generated using the generator for uniform distributions in lnets/tasks/dualnets/distrib/uniform.py and amount to hypercubes embedded in the R^5


# Distribution 1
Distribution 1 is a 5-dimensional uniform distribution, where each individual coordinate is uniform on the interval [-4, -2]; i.e. dim=5, center=-3, supportlength=1

# Distribution 2
Distribution 2 is a 5-dimensional uniform distribution, where each individual coordinate is uniform on the interval [2, 4]; i.e. dim=5, center=3, supportlength=1
