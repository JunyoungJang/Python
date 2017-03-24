def currency_converter(Amount, Exchange_rate):
    return Amount * Exchange_rate

US_dollar_amount = 10000.00
USD_KRW_rate = 1150.08
Korean_won_amount = currency_converter(US_dollar_amount, USD_KRW_rate)

print 'US dollar amount : ', US_dollar_amount
print 'Korean won amount : ', Korean_won_amount

# Exercise:
# Construct a function is_prime that check whether the given integer is prime.

# Exercise:
# Construct a function prime_list_generator that produces the list of prime numbers less than equal to the given integer.

# Exercise:
# Construct a function prime_buster that produces the list of primes forever.
