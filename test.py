class A(int):

    def __init__(self, value, value2):
        self.valuee = value
        self.value2 = value2

    def __new__(cls, value, value2):
        return super(A, cls).__new__(cls, value)


class B(int):

    def __init__(self, value, value2):
        self.valuee = value
        self.value2 = value2

    def __new__(cls, value, value2):
        return super(B, cls).__new__(cls, value)


# Create an instance of A
a = A(1, 2)
b = B(1, 2)

print("a <: B", isinstance(a, B))  # Output will be False
print("b <: A", isinstance(b, A))  # Output will be False
print("int <: A", isinstance(1, A))

# print(str(a))
# print(a.valuee)  # Output will be 1
# print(a.value2)  # Output will be 2
# print(isinstance(a, A))  # Output will be True
# print(isinstance(a, int))  # Output will be True