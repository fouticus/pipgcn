import numpy as np
from configuration import printt


class TrainTest:
    def __init__(self, results_processor=None):
        self.results_processor = results_processor

    def fit_model(self, exp_specs, data, model):
        """
        trains model by iterating minibatches for specified number of epochs
        """
        printt("Fitting Model")
        # train for specified number of epochs
        for epoch in range(1, exp_specs["num_epochs"] + 1):
            self.train_epoch(data["train"], model, exp_specs["minibatch_size"])
        # calculate train and test metrics
        headers, result = self.results_processor.process_results(exp_specs, data, model, "epoch_" + str(epoch))
        # clean up
        self.results_processor.reset()
        model.close()
        return headers, result

    def train_epoch(self, data, model, minibatch_size):
        """
        Trains model for one pass through training data, one protein at a time
        Each protein is split into minibatches of paired examples.
        Features for the entire protein is passed to model, but only a minibatch of examples are passed
        """
        prot_perm = np.random.permutation(len(data))
        # loop through each protein
        for protein in prot_perm:
            # extract just data for this protein
            prot_data = data[protein]
            pair_examples = prot_data["label"]
            n = len(pair_examples)
            shuffle_indices = np.random.permutation(np.arange(n)).astype(int)
            # loop through each minibatch
            for i in range(int(n / minibatch_size)):
                # extract data for this minibatch
                index = int(i * minibatch_size)
                examples = pair_examples[shuffle_indices[index: index + minibatch_size]]
                minibatch = {}
                for feature_type in prot_data:
                    if feature_type == "label":
                        minibatch["label"] = examples
                    else:
                        minibatch[feature_type] = prot_data[feature_type]
                # train the model
                model.train(minibatch)
