import build.pybind_test as pybind_test

print(pybind_test.add(1,2))
print(pybind_test.what)
print(pybind_test.the_answer)

# For OOP functionalities

p = pybind_test.Pet("Molly")
print(p)
print("Pet's name:", p.getName())
p.setName("Bella")
print("Pet's new name:", p.getName())