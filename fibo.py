import matplotlib.pyplot as plt


def fib(f0, f1, N):
	fibo = [f0 , f1]  #initialize 
	   

	for n in range(2,N+1):
		f=fibo[len(fibo)-2]+fibo[len(fibo)-1]
		fibo.append(f)

	print(fibo)
	plt.plot(fibo, label="fib")
	plt.legend()
	plt.show()
	


fib(1,1,20)
	
