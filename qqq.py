# Basic operations

dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
print("Length of dict: ", len(dict))

c="1"
print(c)

dict1 = {'Name': 'Jivin', 'Age': 6};
dict2 = {'Name': 'Pratham', 'Age': 7};
dict3 = {'Name': 'Pranuth', 'Age': 7};
dict4 = {'Name': 'Jivin', 'Age': 6};

# return 0 if dict2==dict1 else (-1 if dict2<dict1 else 1)

print("Return Value: dict1 vs dict2", (dict2 == dict1))
print("Return Value: dict2 vs dict3", (dict2==dict3))
print("Return Value: dict1 vs dict4", (dict1==dict4))

# String representation of dictionary
dict = {'Name': 'Jivin', 'Age': 6}
print("Equivalent String: ", str (dict))

# Copy the dict
dict1 = dict.copy()
print(dict1)

# Create new dictionary with keys from tuple and values to set value
seq = ('name', 'age', 'sex')

dict = dict.fromkeys(seq)
print("New Dictionary: ", str(dict))

dict = dict.fromkeys(seq, 10)
print("New Dictionary: ", str(dict))

# Retrieve value for a given key
dict = {'Name': 'Jivin', 'Age': 6};
print("Value for Age: ", dict.get('Age'))
# Since the key Education does not exist, the second argument will be returned
print("Value for Education: ", dict.get('Education', "First Grade"))

# Check if key in dictionary
print("Age exists? ", dict.get('Age'))
print("Sex exists? ", dict.get('Sex'))

# Return items of dictionary
print("dict items: ", dict.items())

# Return items of keys
print("dict keys: ", dict.keys())

# return values of dict
print("Value of dict: ",  dict.values())

# if key does not exists, then the arguments will be added to dict and returned
print("Value for Age : ", dict.setdefault('Age', None))
print("Value for Sex: ", dict.setdefault('Sex', None))

# Concatenate dicts
dict = {'Name': 'Jivin', 'Age': 6}
dict2 = {'Sex': 'male' }

dict.update(dict2)
print("dict.update(dict2) = ",  dict)
