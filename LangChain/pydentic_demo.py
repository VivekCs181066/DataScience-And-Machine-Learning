from pydantic import BaseModel,EmailStr,Field
from typing import Optional
class Student(BaseModel):
    name:str

new_student={'name':'John'}
# student=Student(new_student) # We will get error in this line because BaseModel expects keyword arguments and we are passing a dictoinary which is considered as positional argument
student=Student(**new_student)
print(student)
# Now lets test with wrong data type
wrong_student={'name':123}
# student=Student(**wrong_student) # We will get error in this line because name expects str but we are passing int
# print(student)
# setting default values
class DefaultStudent(BaseModel):
    name:str ='Vivek'

default_student={}
student=DefaultStudent(**default_student)
print(student)

# Optional fields
class OptionalStudent(BaseModel):
    name: str ='Vivek'
    age:Optional[int]=None
    email:EmailStr
    cgpa :float=  Field(gt=0,lt=10,description="CGPA must be between 0 and 10")
# optional_student={'age':25,'email':'abc@gmail.com','cgpa':'8.5'} in this line you see we have passed cgpa as string but still it got converted into the float .this is the beauty of pPydentic
optional_student={'age':25,'email':'abc@gmail.com','cgpa':5}
student=OptionalStudent(**optional_student)
print(student)