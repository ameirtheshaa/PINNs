def nearest_power_of_2(n):
    """
    Find the nearest number to n that is a power of 2.
    """
    if n < 1:
        return 0
    
    # Find the power of 2 closest to n
    power = 1
    counter = 0
    while power < n:
        power *= 2
        counter += 1

    # Check if the previous power of 2 is closer to n than the current
    if power - n >= n - power // 2:
        return power // 2, counter-1
    else:
        return power, counter



print (nearest_power_of_2(19203129038))