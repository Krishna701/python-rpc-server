# # import main
# class Square:
#    def __init__(self,val):
#       self.val=val
#    def getVal(self):
#       return self.val*self.val
 
 
# class Add_Sub:
#    def add(self, a, b):
#       return a + b
#    def sub(self, a, b):
#       return a - b
 
class NRSService:

   def __init__(self, title, category):
      self.title = title
      self.category = category 

p1  = NRSService("ONlineKhabar", "Politics")

print(p1.title)
print(p1.category)

# class Person:
#   def __init__(self, name, age):
#     self.name = name
#     self.age = age

# p1 = Person("John", 36)

# print(p1.name)
# print(p1.age)
