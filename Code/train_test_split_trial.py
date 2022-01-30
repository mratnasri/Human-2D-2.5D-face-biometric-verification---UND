from sklearn.model_selection import train_test_split
import numpy as np

gray_images_2D = [453,455,457,459,461,463,465,467,303,305,307,309,311,313,315,345,347,349,351,353,355,357,359]
gray_images_25D=[x-1 for x in gray_images_2D]
targets=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3]


mystate=41
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    gray_images_2D, targets, test_size=0.3, shuffle=True, random_state=mystate)
x_val1, x_test1, y_val1, y_test1 = train_test_split(
    x_test1, y_test1, test_size=0.5, shuffle=True, random_state=mystate)

x_train2, x_test2, y_train2, y_test2 = train_test_split(
    gray_images_25D, targets, test_size=0.3, shuffle=True, random_state=mystate)
x_val2, x_test2, y_val2, y_test2 = train_test_split(
    x_test2, y_test2, test_size=0.5, shuffle=True, random_state=mystate)

print("2D")
print(x_train1)
print(y_train1)
print(x_val1)
print(y_val1)
print(x_test1)
print(y_test1)

print("")
print("2.5D")
print(x_train2)
print(y_train2)
print(x_val2)
print(y_val2)
print(x_test2)
print(y_test2)