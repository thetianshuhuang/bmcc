

class NormalWishart:

    likelihood =  # todo
    alloc =  # todo



class MixtureModel:

    def __init__(self, data, assignments, model, components):

        self.data = data
        self.assignments = assignments

        self.model = model.init(self.data)
        self.components = components.init(self.data)

    def gibbs_iter(self):

        call_c_accelerator(
            self.data, self.assignments,
            self.model.capsule,
            self.components.capsule)

