from MyFile import *
# creating object for Square class
object1 = MyFile.Square(5)
print(f"From class Square: {object1.getVal()}")
# creating object for Add_Sub class
object2 = MyFile.Add_Sub()
print(f"From class Add_Sub: Addition {object2.add(2,3)}")
print(f"From class Add_Sub: Subtraction {object2.sub(2,3)}")
 