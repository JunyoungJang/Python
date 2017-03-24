# Python comes with many standard libraries

import math

print math.sqrt(2)
print math.hypot(2,3)   # sqrt(x**2+y**2)
print math.e, math.pi   # as accurate as possible


# Python also provides a help function
import math

help(math)

# And some nicer ways to import
from math import sqrt
print sqrt(3)
# same result
from math import hypot as euclid
print euclid(3,4)
from math import hypot
print hypot(3,4)
from math import * # <- Generally a bad idea Someone cloud add to the library after you start using it


