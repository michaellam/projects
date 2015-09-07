import random
import json

def generate_random_weight():
    weight = random.uniform(-1, 1)
    return weight
    
def dump_json(data, filepath):
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)

def load_json(filepath):
    with open(filepath, 'r') as fp:
        return json.load(fp)
    