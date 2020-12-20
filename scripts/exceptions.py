class EntityOverwriteError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(__self.value__)

class KnowledgeBaseParsingError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class ModelNotFoundError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
