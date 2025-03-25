import random

def generate_random_numbers(filename="musicseed_list.txt", count=100, seed=83):
    random.seed(seed)
    with open(filename, "w") as file:
        for _ in range(count):
            file.write(str(random.randint(10000, 99999)) + "\n")

generate_random_numbers()

