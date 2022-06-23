class Employee:
    num_of_employees = 0
    apply_raise = 1.033

    def __init__(self,first_name,last_name,pay):
        self.first_name = first_name
        self.last_name = last_name
        self.pay = pay
        self.email = first_name + last_name + "." + "@email.com"

        Employee.num_of_employees += 1

    def fullname(self):
        return f"{self.first_name} {self.last_name}"

    def raise_amount(self):
        self.pay = int(self.pay * Employee.apply_raise)

emp_1 = Employee("Krishna","Rijal",10000)
emp_2 = Employee("hello", "user",30000)

print(Employee.num_of_employees)



# print(emp_1.pay)
# emp_1.raise_amount()
# print(emp_1.pay)

# print(emp_1.__dict__)


# emp_1.apply_raise = 1.055
# print(emp_1.__dict__)

# print(Employee.apply_raise)
# print(emp_1.apply_raise)
# print(emp_2.apply_raise)

# print(emp_1.fullname())
# print(emp_2.fullname())

# print(Employee.fullname(emp_1))

# emp_1.first_name = "Krishna"
# emp_1.last_name = "Rijal"
# emp_1.email = "example@gmail.com"
# emp_1.pay = 10000

# emp_2.first_name = "hello"
# emp_2.last_name = "user"
# emp_2.email = "hellouser@gmail.com"
# emp_2.pay = 30000

# print(emp_1.email)
# print(emp_2.email)
# print(f"{emp_1.first_name} {emp_1.last_name}")

