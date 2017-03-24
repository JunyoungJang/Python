# string concatenation using +
a = 'Today'
b = "is"
c = 'Friday'
print a + b + c                  # TodayisFriday
print a + ' ' + b + ' ' + c      # Today is Friday
print '%s %s %s' % (a, b, c)     # Today is Friday

# string concatenation using *
a = 'Today'
print a * 3                      # TodayTodayToday
