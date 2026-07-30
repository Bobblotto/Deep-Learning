"""
def isLeapYear(year):

    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

print(isLeapYear(1600))
"""

num = int(input('Enter a number '))
num2 = int(input('Enter another number '))

print(num&num2)

print(num|num2)

print(~num)

print(num2<<1)

print(num2>>2)

print(1-2//3+4)

# operators of the same priority are worked out from left to right
# exponents are an exception, going from right to left