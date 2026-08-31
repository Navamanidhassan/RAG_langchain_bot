

def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


a = 10
b = 5
print(add(a, b))
print(subtract(a, b))
print(divide(a, b))
