#include "GateMatrix.h"

using namespace qgate;

void set(Matrix2x2C64 &mat,
         const std::complex<double> &a00,
         const std::complex<double> &a01,
         const std::complex<double> &a10,
         const std::complex<double> &a11) {
    mat(0, 0) = a00; mat(0, 1) = a01;
    mat(1, 0) = a10; mat(1, 1) = a11;
}

void mul(const std::complex<double> &coef, Matrix2x2C64 &mat) {
    mat(0, 0) *= coef; mat(0, 1) *= coef;
    mat(1, 0) *= coef; mat(1, 1) *= coef;
}

void assertAllClose(const Matrix2x2C64 &expected, const Matrix2x2C64 &actual) {
    double err = 0.;
    for (int irow = 0; irow < 2; ++irow) {
        for (int icol = 0; icol < 2; ++icol) {
            auto diff = expected(irow, icol) - actual(irow, icol);
            err += (diff * std::conj(diff)).real();
        }
    }
    if (1.e-10 < err) {
        abort();
    }
}


void test_U_gate() {

    const std::complex<double> j(0., 1.);
    Matrix2x2C64 refmat, mat;
    
    /* test_id_U_gate(self) */
    set(refmat, 1, 0, 0, 1);
    U_mat(mat, 0, 0, 0);
    assertAllClose(refmat, mat);
    U1_mat(mat, 0);
    assertAllClose(refmat, mat);

    /* test_id_U2_mat_gate(self)  */
    U_mat(refmat, M_PI / 2., 0, 0);
    U2_mat(mat, 0, 0);
    assertAllClose(refmat, mat);
    
    /* math.pi, 0. */
    U_mat(refmat, M_PI / 2., M_PI, 0);
    U2_mat(mat, M_PI, 0);
    assertAllClose(refmat, mat);
    
    U_mat(refmat, M_PI / 2., - M_PI, 0);
    U2_mat(mat, - M_PI, 0);
    assertAllClose(refmat, mat);
    
    /* M_PI / 2., 0. */
    U_mat(refmat, M_PI / 2., M_PI / 2., 0);
    U2_mat(mat, M_PI / 2., 0.);
    assertAllClose(refmat, mat);

    U_mat(refmat, M_PI / 2., - M_PI / 2., 0);
    U2_mat(mat, - M_PI / 2., 0.);
    assertAllClose(refmat, mat);

    /* M_PI / 4., 0. */
    U_mat(refmat, M_PI / 2., M_PI / 4., 0);
    U2_mat(mat, M_PI / 4., 0.);
    assertAllClose(refmat, mat);

    U_mat(refmat, M_PI / 2., - M_PI / 4., 0);
    U2_mat(mat, - M_PI / 4., 0.);
    assertAllClose(refmat, mat);

    /* 0., M_PI / 2. */
    U_mat(refmat, M_PI / 2., 0., M_PI / 2.);
    U2_mat(mat, 0., M_PI / 2.);
    assertAllClose(refmat, mat);

    U_mat(refmat, M_PI / 2., 0., - M_PI / 2.);
    U2_mat(mat, 0., - M_PI / 2.);
    assertAllClose(refmat, mat);

    /* 0., M_PI / 4. */
    U_mat(refmat, M_PI / 2., 0., M_PI / 4.);
    U2_mat(mat, 0., M_PI / 4.);
    assertAllClose(refmat, mat);

    U_mat(refmat, M_PI / 2., 0., - M_PI / 4.);
    U2_mat(mat, 0., - M_PI / 4.);
    assertAllClose(refmat, mat);
        
    /* test_pauli_x_U_gate */
    set(refmat, 0, 1, 1, 0);
    U_mat(mat, M_PI, 0, M_PI);
    mul(j, mat);
    assertAllClose(refmat, mat);

    set(refmat, 0, - j, j, 0);
    U_mat(mat, M_PI, M_PI / 2., M_PI / 2.);
    mul(j, mat);
    assertAllClose(refmat, mat);
        
    /* test_pauli_z_U_gate */
    set(refmat, 1, 0, 0, -1);
    U_mat(mat, 0., 0., M_PI);
    mul(j, mat);
    assertAllClose(refmat, mat);
        
    U1_mat(mat, M_PI);
    assertAllClose(refmat, mat);

    /* test_hadmard_U_gate(self) */
    double norm = std::sqrt(0.5);
    set(refmat, norm, norm, norm, -norm);
    U_mat(mat, M_PI / 2., 0., M_PI);
    mul(j, mat);
    assertAllClose(refmat, mat);
        
    U2_mat(mat, 0., M_PI);
    mul(j, mat);
    assertAllClose(refmat, mat);
    
}

void test_gate_matrix() {

    const std::complex<double> j(0., 1.);
    const double sqrt_1_2 = std::sqrt(0.5);
    
    /* test_id_U_gate(self) */
    Matrix2x2C64 refmat, mat;
    
    /* test_id_gate */
    set(refmat, 1, 0, 0, 1);
    ID_mat(mat);
    assertAllClose(refmat, mat);
        
    /* test_x_gate */
    set(refmat, 0, 1, 1, 0);
    X_mat(mat);
    assertAllClose(refmat, mat);

    /* test_y_U_gate */
    set(refmat, 0, -j, j, 0);
    Y_mat(mat);
    assertAllClose(refmat, mat);
        
    /* test_z_gate */
    set(refmat, 1, 0, 0, -1);
    Z_mat(mat);
    assertAllClose(refmat, mat);

    /* test_h_gate */
    set(refmat, 1, 1, 1, -1);
    mul(sqrt_1_2, refmat);
    H_mat(mat);
    assertAllClose(refmat, mat);

    /* test_S_gate */
    set(refmat, 1, 0, 0, j);
    S_mat(mat);
    assertAllClose(refmat, mat);

    /* test_Sdg_gate */
    set(refmat, 1, 0, 0, -j);
    adjoint(&mat);
    assertAllClose(refmat, mat);
    
    /* test_T_gate */
    set(refmat, 1, 0, 0, std::complex<double>(sqrt_1_2, sqrt_1_2));
    T_mat(mat);
    assertAllClose(refmat, mat);

    /* test_Tdg_gate */
    set(refmat, 1, 0, 0, std::complex<double>(sqrt_1_2, - sqrt_1_2));
    adjoint(&mat);
    assertAllClose(refmat, mat);
    
    /* test_Rx_gate */
    set(refmat, 1, 0, 0, 1);
    RX_mat(mat, 0);
    assertAllClose(refmat, mat);

    set(refmat, -1, 0, 0, -1);
    RX_mat(mat, M_PI * 2.);
    assertAllClose(refmat, mat);

    set(refmat, 0, -j, -j, 0);
    RX_mat(mat, M_PI);
    assertAllClose(refmat, mat);

    set(refmat, 0, j, j, 0);
    RX_mat(mat, -M_PI);
    assertAllClose(refmat, mat);
        
    set(refmat, 1, -j, -j, 1);
    mul(sqrt_1_2, refmat);
    RX_mat(mat, M_PI / 2.);
    assertAllClose(refmat, mat);
        
    set(refmat, 1, j, j, 1);
    mul(sqrt_1_2, refmat);
    RX_mat(mat, - M_PI / 2.);
    assertAllClose(refmat, mat);
        
    /* test_Ry_gate */
    set(refmat, 1, 0, 0, 1);
    RY_mat(mat, 0);
    assertAllClose(refmat, mat);

    set(refmat, -1, 0, 0, -1);
    RY_mat(mat, M_PI * 2.);
    assertAllClose(refmat, mat);

    set(refmat, 0, -1, 1, 0);
    RY_mat(mat, M_PI);
    assertAllClose(refmat, mat);

    set(refmat, 0, 1, -1, 0);
    RY_mat(mat, -M_PI);
    assertAllClose(refmat, mat);
        
    set(refmat, 1, -1, 1, 1);
    mul(sqrt_1_2, refmat);
    RY_mat(mat, M_PI / 2.);
    assertAllClose(refmat, mat);
    
    set(refmat, 1, 1, -1, 1);
    mul(sqrt_1_2, refmat);
    RY_mat(mat, - M_PI / 2.);
    assertAllClose(refmat, mat);

    /* test_Rz_gate */
    set(refmat, 1, 0, 0, 1);
    RZ_mat(mat, 0);
    assertAllClose(refmat, mat);
    
    set(refmat, -1, 0, 0, -1);
    RZ_mat(mat, M_PI * 2.);
    assertAllClose(refmat, mat);

    set(refmat, -j, 0, 0, j);
    RZ_mat(mat, M_PI);
    assertAllClose(refmat, mat);

    set(refmat, j, 0, 0, -j);
    RZ_mat(mat, -M_PI);
    assertAllClose(refmat, mat);

    set(refmat, 1. - j, 0, 0, 1. + j);
    mul(sqrt_1_2, refmat);
    RZ_mat(mat, M_PI / 2.);
    assertAllClose(refmat, mat);
        
    set(refmat, 1. + j, 0, 0, 1. - j);
    mul(sqrt_1_2, refmat);
    RZ_mat(mat, - M_PI / 2.);
    assertAllClose(refmat, mat);

    /* test_ExpiI_gate */
    set(refmat, 1, 0, 0, 1);
    ExpiI_mat(mat, 0);
    assertAllClose(refmat, mat);

    set(refmat, -1, 0, 0, -1);
    ExpiI_mat(mat, M_PI);
    assertAllClose(refmat, mat);
    
    set(refmat, j, 0, 0, j);
    ExpiI_mat(mat, M_PI / 2.);
    assertAllClose(refmat, mat);
    
    set(refmat, -j, 0, 0, -j);
    ExpiI_mat(mat, -M_PI / 2.);
    assertAllClose(refmat, mat);
    
    set(refmat, 1, 0, 0, 1);
    mul(sqrt_1_2 * (1. + j), refmat);
    ExpiI_mat(mat, M_PI / 4.);
    assertAllClose(refmat, mat);
    
    set(refmat, 1, 0, 0, 1);
    mul(sqrt_1_2 * (1. - j), refmat);
    ExpiI_mat(mat, - M_PI / 4.);
    assertAllClose(refmat, mat);

    /* test_ExpiZ_gate */
    set(refmat, 1, 0, 0, 1);
    ExpiZ_mat(mat, 0);
    assertAllClose(refmat, mat);

    set(refmat, -1, 0, 0, -1);
    ExpiZ_mat(mat, M_PI);
    assertAllClose(refmat, mat);
    
    set(refmat, j, 0, 0, -j);
    ExpiZ_mat(mat, M_PI / 2.);
    assertAllClose(refmat, mat);
    
    set(refmat, -j, 0, 0, j);
    ExpiZ_mat(mat, -M_PI / 2.);
    assertAllClose(refmat, mat);
    
    set(refmat, 1. + j, 0, 0, 1. - j);
    mul(sqrt_1_2, refmat);
    ExpiZ_mat(mat, M_PI / 4.);
    assertAllClose(refmat, mat);
    
    set(refmat, 1. - j, 0, 0, 1. + j);
    mul(sqrt_1_2, refmat);
    ExpiZ_mat(mat, - M_PI / 4.);
    assertAllClose(refmat, mat);

}



int main() {
    test_U_gate();
    test_gate_matrix();
}
