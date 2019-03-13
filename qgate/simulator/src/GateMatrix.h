#pragma once

#include "Types.h"

namespace qgate {

void U_mat(Matrix2x2C64 &mat, double theta, double phi, double lambda);

void U2_mat(Matrix2x2C64 &mat, double phi, double lambda);

void U1_mat(Matrix2x2C64 &mat, double lambda);

void ID_mat(Matrix2x2C64 &mat);

void X_mat(Matrix2x2C64 &mat);

void Y_mat(Matrix2x2C64 &mat);

void Z_mat(Matrix2x2C64 &mat);

void H_mat(Matrix2x2C64 &mat);

void S_mat(Matrix2x2C64 &mat);

void T_mat(Matrix2x2C64 &mat);

void RX_mat(Matrix2x2C64 &mat, double theta);

void RY_mat(Matrix2x2C64 &mat, double theta);

inline
void RZ_mat(Matrix2x2C64 &mat, double phi) {
    U1_mat(mat, phi);
}

void SH_mat(Matrix2x2C64 &mat);

void ExpiI_mat(Matrix2x2C64 &mat, double theta);

void ExpiZ_mat(Matrix2x2C64 &mat, double theta);

void adjoint(Matrix2x2C64 *mat);

}
