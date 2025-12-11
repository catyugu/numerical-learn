import build.pybind_test as pybind_test

print(pybind_test.add(1,2))
print(pybind_test.what)
print(pybind_test.the_answer)

# For OOP functionalities

p = pybind_test.Pet("Molly")

# The print(p) prints meaningful info if and only if we define the Pet class with __repr__
print(p)
print("Pet created. Its name is:", p.name)
print("Pet's name:", p.getName())
p.setName("Bella")
print("Pet's new name:", p.getName())

# This works if and only if we define the Pet class with dynamic_attr
# It works just like a native Python class. Whoa!
p.age = 1
print("Pet's age:", p.age)