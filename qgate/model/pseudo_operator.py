from .model import Operator

class ClauseBegin(Operator) :
    def copy(self) :  # return self because no properties
        return self

class ClauseEnd(Operator) :
    def copy(self) :  # return self because no properties
        return self

