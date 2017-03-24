import Greeting

g = Greeting.Greeter('Fred')  # Construct an instance of the Greeter class
g.name = 'Paul'
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
print g._city
del g
