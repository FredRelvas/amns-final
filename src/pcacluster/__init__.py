"""PCA before clustering: shared interfaces of the group project."""

# One seed per repetition, passed to everything random in it: synthetic data
# generation or subsampling, train/test split and clustering initialization.
SEEDS = tuple(range(10))
