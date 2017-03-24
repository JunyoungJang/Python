purse = dict()
purse['money'] = 12
purse['candy'] = 3
purse['tissues'] = 75
purse['candy'] = purse['candy'] + 2
del purse['candy']
print purse # {'money': 12, 'tissues': 75}
