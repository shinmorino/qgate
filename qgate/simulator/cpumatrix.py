import qgate.model.gate_type as gtype
from . import glue

gtype.U.cmatf = glue.register_matrix_factory('U')
gtype.U2.cmatf = glue.register_matrix_factory('U2')
gtype.U1.cmatf = glue.register_matrix_factory('U1')
gtype.ID.cmatf = glue.register_matrix_factory('ID')
gtype.X.cmatf = glue.register_matrix_factory('X')
gtype.Y.cmatf = glue.register_matrix_factory('Y')
gtype.Z.cmatf = glue.register_matrix_factory('Z')
gtype.H.cmatf = glue.register_matrix_factory('H')
gtype.S.cmatf = glue.register_matrix_factory('S')
gtype.T.cmatf = glue.register_matrix_factory('T')
gtype.RX.cmatf = glue.register_matrix_factory('RX')
gtype.RY.cmatf = glue.register_matrix_factory('RY')
gtype.RZ.cmatf = glue.register_matrix_factory('RZ')
gtype.ExpiI.cmatf = glue.register_matrix_factory('ExpiI')
gtype.ExpiZ.cmatf = glue.register_matrix_factory('ExpiZ')
# Utility
gtype.SH.cmatf = glue.register_matrix_factory('SH')
