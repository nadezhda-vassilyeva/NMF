import numpy as np
from sklearn.utils import shuffle


from torch.utils.data import Dataset


class Dataloader():
    def __init__(self, V, n_negatives, n_batches, from_entities, valid):
        self.from_entities = from_entities
        row_inds, col_inds = V.nonzero()
        self.data = np.array([(i, j) for i, j in zip(row_inds, col_inds)])
        self.batchsize = np.int_(np.ceil(len(self.data) / n_batches))
        self.n_negatives = n_negatives
        self.validation_set = []
        self.highest = V.shape[1] if from_entities else V.shape[0]
        self.valid = valid

    def construct_batches(self):
        self.data = shuffle(self.data)

        batches = [self.data[i : i+self.batchsize] for i in range(0, len(self.data), self.batchsize)]

        already_sampled = set(self.validation_set)
        bcount = 0
        index = 0 if self.from_entities else 1

        for batch in batches:
            batch_counter = Counter([b[index] for b in batch])
            negative_samples = self.rejection_sampling(batch_counter, already_sampled)
            already_sampled |= negative_samples
            batch.extend(list(negative_samples))

        # isolate validation set, if needed
        if self.valid and not self.validation_set:
            self.validation_set = batches[0]
            batches = batches[1:]

        return batches


    def rejection_sampling(self, batch_counter, already_sampled):
        total_negative_sample = set()       # negative sample for the whole batch
        for entity, n_occurences in batch_counter.items():
            size = n_occurences * self.n_negatives    # size of negative sample
            # do the actual sampling
            negative_sample_per_element = set()     # keeps track of negative sample for this element in batch
            while len(negative_sample_per_element) < size:
                n_items = size - len(negative_sample_per_element)   # number of negative samples we should try getting
                random_sample = set(np.random.randint(low = 0, high = high, size = n_items))    # sample words (columns)
                if self.from_entities:
                    random_sample = set([(entity, rs) for rs in random_sample]) - nonzeros - already_sampled   # create tuples, remove all nonzero elements
                else:
                    random_sample = set([(rs, entity) for rs in random_sample]) - nonzeros - already_sampled   # same, but for words
                negative_sample_for_element |= random_sample
            total_negative_sample |= negative_sample_for_element    # add constructed negative sample to the set of negative samples for this batch
        return total_negative_sample

