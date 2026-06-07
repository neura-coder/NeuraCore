import json
import random

templates = [
    ("Write a function that adds two numbers.", "def add(a, b):\n    return a + b"),
    ("Create a Python function to add two integers.", "def add(a, b):\n    return a + b"),
    ("Implement addition of two numbers in Python.", "def add(a, b):\n    return a + b"),
    ("Write a function called add that returns sum of a and b.", "def add(a, b):\n    return a + b"),
    ("Define a function to add two values.", "def add(a, b):\n    return a + b"),
]

data = []
for i in range(5000):
    prompt, code = random.choice(templates)
    data.append({"code": prompt + "\n" + code})

with open("data/train.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved 5000 samples to data/train.json")
