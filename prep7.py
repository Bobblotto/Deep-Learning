"""nums = []

evens = 0
odds = 0

while True:
    num = int(input('Enter a number '))

    if num == 0:
        break
    elif num%2 == 0:
        evens += 1
    else:
        odds += 1

    print(f'even: {evens}')
    print(f'odd: {odds}')"""

"""nums = [5, 3, 4, 8, 6, 0, 4, 7, 9]

evens = 0
odds = 0

for num in nums:

    if num == 0:
        break
    elif num%2 == 0:
        evens += 1
    else:
        odds += 1

print(f'even: {evens}')
print(f'odd: {odds}')"""

def removeVowels(word):
    newWord = ''

    for i in word:
        if i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u':
            continue
        newWord += i

    return newWord

print(removeVowels('Hello world'))