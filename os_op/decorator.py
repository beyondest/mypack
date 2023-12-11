import time

def timing(func):
    def inner(*args, **kwargs):
        t1=time.perf_counter()
        result=func(*args, **kwargs)
        t2=time.perf_counter()
        elapsed_time=t2-t1
        return [result,elapsed_time]
    return inner
