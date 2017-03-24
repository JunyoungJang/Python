# From Hours and Rate, compute Pay. Over time rate (above 40 hours) is 1.5 times rate.

def pay(Hours, Rate):
    Pay = min(Hours, 40) * Rate + max(Hours - 40, 0) * (1.5 * Rate)
    return Pay

Hours = raw_input('Enter Hours : ')
Rate = raw_input('Enter Rate : ')

Hours = float(Hours)
Rate = float(Rate)

Pay = pay(Hours, Rate)

print 'Hours :', Hours
print 'Rate :', Rate
print 'Overtime Rate (over 40 hours) :', 1.5 * Rate
print 'Pay :', Pay


