print("Hello word")

a = 5
print(a)
print(type(a))

a = 5.0
print(a)
print(type(a))

l = [1, 2, 3, 4]
d = {"pepe": 12}

print(l)
print(d)

print(d["pepe"])

if d["pepe"] % 2 == 0:
    print("pepe es par")

else:
    print("pepe es impar")

i = 0;
while i < 5:
    print(i)
    i = i + 1

print("=========")
for i, e in enumerate(l):
    print(f"{i}: {e}")

print("=========")
for i in range(10):
    print(i)