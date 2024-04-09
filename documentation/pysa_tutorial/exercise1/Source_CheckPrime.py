def takeInput():
    
    u = input("")
    return u

def checkPrime(num):
    prime = False
    flag = False
    ret = 0
    if num == 1:
        prime = False
    elif num > 1:
        for i in range(2, num):
            ret = num%i
            if (num % i) == 0:
                # if factor is found, set flag to True
                flag = True
                # break out of loop
                break
    
        # check if flag is True
        if flag:
            prime=False
        else:
            prime = True
    return ret

def printInput(n):
    print(n)
    
    
def main():
    i = takeInput()
    p = checkPrime(i)
    printInput(p)
    
    