import math

def epochs_to_iterations(epochs = 1, batch_size = 64, dataset_size = 84):
    iterations_per_epoch = dataset_size/batch_size
    print(f"The number of iterations per epoch is {iterations_per_epoch}")
    return epochs*math.ceil(iterations_per_epoch)

def iterations_to_epochs(iterations = 1, batch_size = 64, dataset_size = 84):
    iterations_per_epoch = dataset_size/batch_size
    print(f"The number of epochs per iteration is {iterations_per_epoch}")
    return iterations/math.ceil(iterations_per_epoch)

if __name__ == "__main__":
    result = epochs_to_iterations()
    print(result)