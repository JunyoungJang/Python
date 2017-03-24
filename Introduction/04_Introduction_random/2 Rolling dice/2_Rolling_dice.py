import random

N_STEPS = 10 # number of steps in simulation

dice_roll_results = []
for step in range(N_STEPS):
    uniform_dice = random.uniform(0, 1)
    print uniform_dice
    if uniform_dice >  5./6.:
        dice_roll_results.append(6)
    elif uniform_dice > 4./6.:
        dice_roll_results.append(5)
    elif uniform_dice > 3./6.:
        dice_roll_results.append(4)
    elif uniform_dice > 2./6.:
        dice_roll_results.append(3)
    elif uniform_dice > 1./6.:
        dice_roll_results.append(2)
    else:
        dice_roll_results.append(1)

print dice_roll_results