#define _USE_MATH_DEFINES  /* to define M_PI on Windows. */
#include <math.h>
#include "GateMatrix.h"

using namespace qgate;

namespace {

std::complex<double> j(0., 1.);

}

void qgate::U_mat(Matrix2x2C64 &mat, double theta, double phi, double lambda) {
    /* Global phase is adjusted for consistency with qgate gate set.
       The difference from OpenQASM definition is exp(i (lambda + phi) / 2). */
    double theta2 = theta * 0.5;
    double cos_theta_2 = std::cos(theta2);
    double sin_theta_2 = std::sin(theta2);
    double lambda2 = lambda * 0.5;
    double phi2 = phi * 0.5;

    mat(0, 0) =   std::exp(j * (-lambda2 - phi2)) * cos_theta_2;
    mat(0, 1) = - std::exp(j * ( lambda2 - phi2)) * sin_theta_2;
    mat(1, 0) =   std::exp(j * (-lambda2 + phi2)) * sin_theta_2;
    mat(1, 1) =   std::exp(j * ( lambda2 + phi2)) * cos_theta_2;
}

void qgate::U2_mat(Matrix2x2C64 &mat, double phi, double lambda) {
    /* Global phase is adjusted for consistency with qgate gate set.
       The difference from OpenQASM definition is exp(i (lambda + phi) / 2). */
    double norm = std::sqrt(0.5);
    double lambda2 = lambda * 0.5;
    double phi2 = phi * 0.5;
    mat(0, 0) =   std::exp(j * (-lambda2 - phi2)) * norm;
    mat(0, 1) = - std::exp(j * ( lambda2 - phi2)) * norm;
    mat(1, 0) =   std::exp(j * (-lambda2 + phi2)) * norm;
    mat(1, 1) =   std::exp(j * ( lambda2 + phi2)) * norm;
}

void qgate::U1_mat(Matrix2x2C64 &mat, double lambda) {
    /* Global phase is not modified from OpenQASM definition.
       U1 is a phase shift gate, and phase shift is specified by lambda. */
    mat(0, 0) =   1.;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   std::exp(j * lambda);
}

void qgate::ID_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   1.;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   1.;
}

void qgate::X_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   0.;
    mat(0, 1) =   1.;
    mat(1, 0) =   1.;
    mat(1, 1) =   0.;
}

void qgate::Y_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   0.;
    mat(0, 1) = - j;
    mat(1, 0) =   j;
    mat(1, 1) =   0.;
}

void qgate::Z_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   1.;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) = - 1.;
}

void qgate::H_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   1. * std::sqrt(0.5);
    mat(0, 1) =   1. * std::sqrt(0.5);
    mat(1, 0) =   1. * std::sqrt(0.5);
    mat(1, 1) = - 1. * std::sqrt(0.5);
}

void qgate::S_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   1.;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   j;
}

void qgate::T_mat(Matrix2x2C64 &mat) {
    mat(0, 0) =   1.;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   std::exp(M_PI * 0.25 * j);
}

void qgate::RX_mat(Matrix2x2C64 &mat, double theta) {
    double theta2 = theta / 2.;
    double cos_theta_2 = std::cos(theta2);
    double sin_theta_2 = std::sin(theta2);
    mat(0, 0) =       cos_theta_2;
    mat(0, 1) = - j * sin_theta_2;
    mat(1, 0) = - j * sin_theta_2;
    mat(1, 1) =       cos_theta_2;
}

void qgate::RY_mat(Matrix2x2C64 &mat, double theta) {
    double theta2 = theta / 2.;
    double cos_theta_2 = std::cos(theta2);
    double sin_theta_2 = std::sin(theta2);
    mat(0, 0) =   cos_theta_2;
    mat(0, 1) = - sin_theta_2;
    mat(1, 0) =   sin_theta_2;
    mat(1, 1) =   cos_theta_2;
}

void qgate::RZ_mat(Matrix2x2C64 &mat, double theta) {
    double theta2 = theta / 2.;
    mat(0, 0) =   std::exp(-j * theta2);
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   std::exp( j * theta2);
}

void qgate::ExpiI_mat(Matrix2x2C64 &mat, double theta) {
    std::complex<double> d(std::exp(theta  * j));
    mat(0, 0) =   d;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   d;
}

void qgate::ExpiZ_mat(Matrix2x2C64 &mat, double theta) {
    std::complex<double> d0(std::exp(theta  * j));
    std::complex<double> d1(std::exp(- theta  * j));
    mat(0, 0) =   d0;
    mat(0, 1) =   0.;
    mat(1, 0) =   0.;
    mat(1, 1) =   d1;
}

void qgate::SH_mat(Matrix2x2C64 &mat) {
    double norm = std::sqrt(0.5);
    mat(0, 0) =   1. * norm;
    mat(0, 1) =   1. * norm;
    mat(1, 0) =   j  * norm;
    mat(1, 1) = - j  * norm;
}

void qgate::adjoint(Matrix2x2C64 *_mat) {
    Matrix2x2C64 &mat = *_mat;
    mat(0, 0) = std::conj(mat(0, 0));
    qgate::ComplexType<double> tmp = mat(0, 1);
    mat(0, 1) = std::conj(mat(1, 0));
    mat(1, 0) = std::conj(tmp);
    mat(1, 1) = std::conj(mat(1, 1));
}
