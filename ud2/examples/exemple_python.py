#!/usr/bin/env python

# Python Basics Example Script

# 1. Variables and Data Types
# ----------------------------

# Integer
age = 25
# Float
height = 5.9
# String
name = "Alice"
# Boolean
is_student = True

print(f"My name is {name}, I am {age} years old, {height} feet tall, and it is {is_student} that I am a student.")

# 2. Conditional Statements (if-elif-else)
# ----------------------------------------

if age < 18:
    print("You are a minor.")
elif age < 65:
    print("You are an adult.")
else:
    print("You are a senior.")

# 3. Loops (for, while)
# ----------------------

# for loop example
print("\nFor loop example:")
for i in range(5):  # Will loop from 0 to 4
    print(f"Iteration {i}")

# while loop example
print("\nWhile loop example:")
count = 0
while count < 3:
    print(f"Count is {count}")
    count += 1

# 4. Functions
# -------------

def greet(name, is_morning=True):
    """Function to greet a person based on the time of day."""
    if is_morning:
        return f"Good morning, {name}!"
    else:
        return f"Hello, {name}!"

print("\nFunction example:")
print(greet("Alice"))
print(greet("Bob", False))

# 5. Lists and Dictionaries
# ---------------------------

# List of numbers
numbers = [10, 20, 30, 40, 50]

# Dictionary representing a person
person = {
    "name": "John",
    "age": 30,
    "is_student": False
}

# Accessing list and dictionary elements
print("\nList and Dictionary example:")
print(f"The first number in the list is {numbers[0]}")
print(f"The person's name is {person['name']} and they are {person['age']} years old.")

# 6. Classes and Objects
# -----------------------

class Dog:
    """A simple class representing a dog."""

    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        return f"{self.name} says Woof!"

# Creating an instance (object) of the Dog class
my_dog = Dog("Buddy", "Golden Retriever")

print("\nClass and Object example:")
print(f"My dog's name is {my_dog.name} and he is a {my_dog.breed}.")
print(my_dog.bark())