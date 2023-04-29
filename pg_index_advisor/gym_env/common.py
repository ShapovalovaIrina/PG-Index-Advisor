from enum import Enum


class EnvironmentType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3

class IndexType(Enum):
    REAL = 1
    VIRTUAL = 2
