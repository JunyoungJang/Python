def threshold(signal):
    return 1.0/sum(signal)

t = threshold
print t([0.1, 0.4, 0.2])
