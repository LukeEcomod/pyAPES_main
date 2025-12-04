from multiprocessing import Queue, Process

def worker(queue, num):
    queue.put(f"Message from worker {num}")

if __name__ == '__main__':
    queue = Queue()
    processes = []

    for i in range(5):
        process = Process(target=worker, args=(queue, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    while not queue.empty():
        print(queue.get())