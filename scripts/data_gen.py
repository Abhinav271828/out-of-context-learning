from tqdm import tqdm
import random

"""
We have six regular languages, of which three have the alphabet {a, b} and three have the alphabet {a, b, c}.
The languages are:
- Σ = {a, b}
    - a*b*
    - (a|bb)*a
    - (ab*a)*
- Σ = {a, b, c}
    - a*bc*
    - ((a|b)c)*
    - (ab[c])*
"""

MAX_LENGTH = 20

def generate_L1(filename, num_samples):
    """
    Generate num_samples samples of the language L1 = a*b*.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            r = random.randint(1, MAX_LENGTH)
            sample = "a" * r + "b" * random.randint(0, MAX_LENGTH-r)
            f.write(sample + "\n")

def generate_L2(filename, num_samples):
    """
    Generate num_samples samples of the language L2 = (a|bb)*a.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            if random.random() < 0.5:
                sample = "a"
            else:
                sample = "bb"
            while len(sample) < MAX_LENGTH - 1:
                r = random.random()
                if r < 0.33:
                    sample += "a"
                elif r < 0.66:
                    sample += "bb"
                else:
                    break
            sample += "a"
            f.write(sample + "\n")  

def generate_L3(filename, num_samples):
    """
    Generate num_samples samples of the language L3 = (ab*a)*.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            sample = "a"
            r = random.gauss(3, 2)
            sample += "b" * int(r)
            sample += "a"
            while len(sample) < MAX_LENGTH:
                if random.random() > 0.4: break
                sample += "a"
                r = random.gauss(3, 2)
                sample += "b" * int(r)
                sample += "a"
                f.write(sample + "\n")

def generate_L4(filename, num_samples):
    """
    Generate num_samples samples of the language L4 = a*bc*.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            r = random.randint(1, MAX_LENGTH-1)
            sample = "a" * r + "b" + "c" * random.randint(0, MAX_LENGTH-1-r)
            f.write(sample + "\n")

def generate_L5(filename, num_samples):
    """
    Generate num_samples samples of the language L5 = ((a|b)c)*.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            l = random.randint(1, MAX_LENGTH)
            sample = ""
            while len(sample) < l:
                if random.random() < 0.5:
                    sample += "a"
                else:
                    sample += "b"
                sample += "c"
            f.write(sample + "\n")

def generate_L6(filename, num_samples):
    """
    Generate num_samples samples of the language L6 = (ab[c])*.
    """
    with open('../data/' + filename, "w") as f:
        for _ in tqdm(range(num_samples)):
            l = random.randint(1, MAX_LENGTH)
            sample = ""
            while len(sample) < l:
                sample += "a"
                sample += "b"
                if random.random() < 0.5:
                    sample += "c"
            f.write(sample + "\n")

def create_data_for(language, num_samples):
    eval(f'generate_L{language}')(f'L{language}.txt', num_samples)

#create_data_for(1, 1000000)
#create_data_for(2, 1000000)
#create_data_for(3, 1000000)
#create_data_for(4, 1000000)
#create_data_for(5, 1000000)
#create_data_for(6, 1000000)