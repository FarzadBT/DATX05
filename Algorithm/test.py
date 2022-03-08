import multiprocessing as mp

def square(x):
    return x*x

def power_n(x, n=2):
    return x ** n

def main():
    pool = mp.Pool(processes=mp.cpu_count())

    result = pool.map(square, range(20))
    print(result)

    result = pool.starmap(power_n, [(x, 3) for x in range(20)])
    print(result)

if __name__ == "__main__":
    main()