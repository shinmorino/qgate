import qgate.model.gate as gate
from . import glue

gate.U.cmatf = glue.register_matrix_factory('U')
gate.U2.cmatf = glue.register_matrix_factory('U2')
gate.U1.cmatf = glue.register_matrix_factory('U1')
gate.ID.cmatf = glue.register_matrix_factory('ID')
gate.X.cmatf = glue.register_matrix_factory('X')
gate.Y.cmatf = glue.register_matrix_factory('Y')
gate.Z.cmatf = glue.register_matrix_factory('Z')
gate.H.cmatf = glue.register_matrix_factory('H')
gate.S.cmatf = glue.register_matrix_factory('S')
gate.T.cmatf = glue.register_matrix_factory('T')
gate.RX.cmatf = glue.register_matrix_factory('RX')
gate.RY.cmatf = glue.register_matrix_factory('RY')
# RZ is an alias of U1.
