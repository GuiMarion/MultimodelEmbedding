import matplotlib.pyplot as plt

f = [1 , 1]  #initialize
N = 10 
fibo = f       

for n in range(2,N+1):
    f = [f[1],f[0]+f[1]]
    fibo.append(f[1])

print(fibo)
plt.plot(fibo, label="fib")
plt.legend()
plt.show()