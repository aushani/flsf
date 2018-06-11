import threading
import Queue
from old_filter_sample import *

class OldFilterBatchManager:

    def __init__(self, filter_data, batch_size = 1):
        self.data_manager = filter_data
        self.batch_size = batch_size

        self.data_queue = Queue.Queue(maxsize = 100)

        self.load_threads = []

        for i in range(4):
            self.load_thread = threading.Thread(target=self.run)
            self.load_thread.daemon = True
            self.load_thread.start()

    def run(self):
        while True:
            filter_samples = self.data_manager.get_next_samples(self.batch_size)
            old_samples = OldFilterSampleSet(filter_samples)
            self.data_queue.put(old_samples)

    def get_next_batch(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item
