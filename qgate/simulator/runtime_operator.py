import qgate.model as model

class Observable :
    pass

# deined as a base class
class Observer :
    pass

    #def wait(self) :
    #    pass

    #@property
    #def observed(self) :
    #    return None
    
    #def get_value(self) :
    #    pass
    

class ControlledGate :
    def __init__(self, qstates, control_lanes, gate_type, adjoint, target_lane) :
        self.qstates = qstates
        self.control_lanes = control_lanes
        self.gate_type = gate_type
        self.adjoint = adjoint
        self.target_lane = target_lane

class Gate :
    def __init__(self, qstates, gate_type, adjoint, lane) :
        self.qstates = qstates
        self.gate_type = gate_type
        self.adjoint = adjoint
        self.lane = lane
        
class Reset :
    def __init__(self, qstates, lane) :
        self.qstates = qstates
        self.lane = lane
        
class ReferencedObservable(Observable) :

    def set_observer(self, obs) :
        self.observer = obs

    def set(self, value) :
        self.observer.set_value(value)
        
    def get_observer(self) :
        return self.observer
    
class MeasureZ(ReferencedObservable) :
    def __init__(self, randnum, qstates, lane) :
        self.randnum = randnum
        self.qstates = qstates
        self.lane = lane
        
class Prob(ReferencedObservable) :
    def __init__(self, qstates, lane) :
        self.qstates = qstates
        self.lane = lane

