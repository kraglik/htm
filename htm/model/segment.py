class ApicalSegment:
    def __init__(self):
        self.synapses = []


class DistalSegment:
    def __init__(self, cell):
        self.cell = cell
        self.learning = False
        self.was_learning = False
        self.active = False
        self.was_active = False
        self.sequence = False
        self.synapses = []

    def step(self):
        self.was_learning = self.learning
        self.learning = False
        self.was_active = self.active
        self.active = False
        self.sequence = False


class ProximalSegment:
    def __init__(self):
        self.synapses = []

    def increase_active_permanences(self, increase, limit):
        for synapse in self.synapses:
            if synapse.is_active:
                synapse.permanence = min(
                    limit,
                    synapse.permanence + increase
                )

    def decrease_inactive_permanences(self, decrease):
        for synapse in self.synapses:
            if synapse.is_active:
                synapse.permanence = max(
                    0.0,
                    synapse.permanence - decrease
                )

