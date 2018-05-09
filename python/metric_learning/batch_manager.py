from data import *
import threading
import Queue

class BatchManager:

    def __init__(self, data_manager, batch_size = 1):
        self.data_manager = data_manager
        self.batch_size = batch_size

        self.data_queue = Queue.Queue(maxsize = 10)

        self.load_thread = threading.Thread(target=self.run)
        self.load_thread.daemon = True
        self.load_thread.start()

    def run(self):
        while True:
            self.data_queue.put(self.data_manager.get_next_samples(self.batch_size))

    def get_next_batch(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item
