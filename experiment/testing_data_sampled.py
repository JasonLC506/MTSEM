import random

from experiment import DataGeneratorTrainTest


def sample(
        feature_file,
        label_file,
        weight_file=None,
        task_file=None,
        sample_rate=0.1
):
    inputs = []
    outputs = []
    output_pattern = "_sampled_%d" % int(100 * sample_rate)
    ff = open(feature_file, "r")
    inputs.append(ff)
    outputs.append(feature_file + output_pattern)
    lf = open(label_file, "r")
    inputs.append(lf)
    outputs.append(label_file + output_pattern)
    if weight_file is not None:
        wf = open(weight_file, 'r')
        inputs.append(wf)
        outputs.append(weight_file + output_pattern)
    if label_file is not None:
        tf = open(task_file, 'r')
        inputs.append(tf)
        outputs.append(task_file + output_pattern)
    outputs = list(map(lambda x: open(x, "w"), outputs))
    cnt = 0
    while True:
        lines = list(map(lambda x: x.readline(), inputs))
        if lines[0] == "":
            break
        rnd = random.random()
        if rnd <= sample_rate:
            for i in range(len(lines)):
                outputs[i].write(lines[i])
            cnt += 1
        else:
            pass
    print("sampled size: %d" % cnt)
    for i in range(len(inputs)):
        inputs[i].close()
        outputs[i].close()


class StageWiseSample(DataGeneratorTrainTest):
    def __init__(
            self,
            feature_file,
            label_file,
            task_file,
            sample_rate=0.2,
            prefix=""
    ):
        super(StageWiseSample, self).__init__(
            feature_file=feature_file,
            label_file=label_file,
            task_file=task_file,
            train_ratio=sample_rate,
            valid_ratio=0.0
        )
        self.sample_rate = sample_rate
        # self.output_pattern = "_sampled_%d" % int(100 * sample_rate)
        # self.output_pattern_remain = "_remained_%d" % int(100 * (1 - sample_rate))
        self.output_pattern = "_test" + prefix
        self.output_pattern_remain = "_train" + prefix
        self.inputs = [feature_file, label_file, task_file]
        self.outputs = self.outputs_remain = []

    def sample(self):
        sample_index, _a, _b = self._train_test_index_split(
            train_ratio=self.sample_rate,
            valid_ratio=0.0,
            stage_wise=True
        )
        sample_index = sorted(sample_index.tolist())
        # write into files #
        self.outputs = list(map(lambda x: x + self.output_pattern, self.inputs))
        self.outputs_remain = list(map(lambda x: x + self.output_pattern_remain, self.inputs))
        input_files = list(map(lambda x: open(x, 'r'), self.inputs))
        self.outputs = list(map(lambda x: open(x, 'w'), self.outputs))
        self.outputs_remain = list(map(lambda x: open(x, 'w'), self.outputs_remain))
        for cnt in range(self.data_size):
            lines = list(map(lambda x: x.readline(), input_files))
            if len(sample_index) > 0 and cnt == sample_index[0]:
                sample_index.pop(0)
                for i in range(len(lines)):
                    self.outputs[i].write(lines[i])
            else:
                for i in range(len(lines)):
                    self.outputs_remain[i].write(lines[i])
        for i in range(len(input_files)):
            input_files[i].close()
            self.outputs[i].close()
            self.outputs_remain[i].close()
        self.outputs = list(map(lambda x: x + self.output_pattern, self.inputs))
        self.outputs_remain = list(map(lambda x: x + self.output_pattern_remain, self.inputs))


if __name__ == "__main__":
    data_dir = "../data/SEM/"
    sampler = StageWiseSample(
        feature_file=data_dir + "feature_train",
        label_file=data_dir + "label_train",
        task_file=data_dir + "id_train",
        sample_rate=0.2
    )
    sampler.sample()
