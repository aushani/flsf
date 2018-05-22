import threading
import Queue
from old_flow_sample import *

class OldBatchManager:

    def __init__(self, flow_data, batch_size = 1):
        self.data_manager = flow_data
        self.batch_size = batch_size

        self.data_queue = Queue.Queue(maxsize = 10)

        self.load_thread = threading.Thread(target=self.run)
        self.load_thread.daemon = True
        self.load_thread.start()

    def run(self):
        while True:
            flow_samples = self.data_manager.get_next_samples(self.batch_size)
            old_samples = OldFlowSampleSet(flow_samples)
            self.data_queue.put(old_samples)

    def get_next_batch(self):
        item = self.data_queue.get()
        self.data_queue.task_done()

        return item
