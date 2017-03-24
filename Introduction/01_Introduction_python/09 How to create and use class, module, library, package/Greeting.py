class Greeter(object):
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable
        self._city = 'Seoul'
    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name
    # Destructor
    def __del__(self):
        print "Objects generated using class Greeter destructed!"