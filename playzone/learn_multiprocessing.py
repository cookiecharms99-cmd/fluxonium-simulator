import time

import math

import multiprocessing as mp


def worker(conn):
    for i in range(5):
        conn.send(i * i)

    conn.close()

if __name__ == "__main__":
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target = worker, args = (child_conn, ))
    p.start()

    for _ in range(5):
        print(f"Received: {parent_conn.recv()}")

    p.join()

    
    