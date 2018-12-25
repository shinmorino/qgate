#include "DeviceTypes.h"
#include "DeviceQubitStates.h"
#include <string.h>
#include <algorithm>

using namespace qgate_cuda;
using qgate::Qone;
using qgate::Qtwo;


template class DeviceQubitStates<float>;
template class DeviceQubitStates<double>;
