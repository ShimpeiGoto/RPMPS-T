// Licensed under the MIT License <http://opensource.org/MIT>
//
// Copyright (c) 2020 Shimpei Goto
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef UUID_95124DA6_52D9_4AE3_8F5D_2ACA5ECDC5C5
#define UUID_95124DA6_52D9_4AE3_8F5D_2ACA5ECDC5C5
#include <itensor/all_mps.h>
#include <unordered_map>
#include <utility>

namespace ZigZag_Trotter {
        class ZigZag_Bond{
                private:
                        int N_;
                        double J_, J2_, Hz_;
                        itensor::SiteSet sites_;
                        std::unordered_map<size_t, itensor::ITensor> swap_gates_;
                        std::unordered_map<size_t, std::pair<itensor::ITensor, itensor::ITensor>> spectrum_;

                public:
                        ZigZag_Bond(int N, double J, double J2, const itensor::SiteSet &sites) : N_(N), J_(J), J2_(J2), Hz_(0.0), sites_(sites) {}
                        ZigZag_Bond(int N, double J, double J2, double Hz, const itensor::SiteSet &sites) : N_(N), J_(J), J2_(J2), Hz_(Hz), sites_(sites) {}
                        itensor::ITensor BondTerm(size_t idx1, size_t idx2, std::complex<double> tau, size_t site_idx);
                        itensor::ITensor Swap(size_t site_idx);
        };

        itensor::ITensor ZigZag_Bond::BondTerm(size_t idx1, size_t idx2, std::complex<double> tau, size_t site_idx) {
                if (spectrum_.find((N_+1)*idx1 + idx2) == spectrum_.end()) {
                        auto bond_term = itensor::ITensor(itensor::dag(sites_(site_idx)), itensor::dag(sites_(site_idx+1)), itensor::prime(sites_(site_idx)), itensor::prime(sites_(site_idx+1)));
                        bool nearest = ((idx1 > idx2 and idx1 - idx2 == 1) or (idx2 > idx1 and idx2 - idx1 == 1));
                        double J_now = nearest ? J_ : J2_;
                        bond_term += 0.5 * J_now * itensor::op(sites_, "S+", site_idx) * itensor::op(sites_, "S-", site_idx+1);
                        bond_term += 0.5 * J_now * itensor::op(sites_, "S-", site_idx) * itensor::op(sites_, "S+", site_idx+1);
                        bond_term += J_now * itensor::op(sites_, "Sz", site_idx) * itensor::op(sites_, "Sz", site_idx+1);
                        if (nearest) {
                                double H1 = 0.5*Hz_, H2 = 0.5*Hz_;
                                if (idx1 == 1) {
                                        H1 *= 2.0;
                                }
                                if (idx2 == N_) {
                                        H2 *= 2.0;
                                }
                                bond_term += H1 * itensor::op(sites_, "Sz", site_idx) * itensor::op(sites_, "Id", site_idx+1);
                                bond_term += H2 * itensor::op(sites_, "Id", site_idx) * itensor::op(sites_, "Sz", site_idx+1);
                        }
                        itensor::ITensor U, d;
                        itensor::diagHermitian(bond_term, U, d);
                        spectrum_[(N_+1)*idx1 + idx2] = {U, d};
                        if (tau.imag() == 0.){
                                d.apply([tau](double x){return std::exp(tau.real()*x);});
                        }
                        else {
                                d.apply([tau](double x){return std::exp(tau*x);});
                        }

                        return itensor::prime(U)*d*itensor::dag(U);
                }

                auto [U, d] = spectrum_[(N_+1)*idx1 + idx2];
                if (tau.imag() == 0.){
                        d.apply([tau](double x){return std::exp(tau.real()*x);});
                }
                else {
                        d.apply([tau](double x){return std::exp(tau*x);});
                }
                return itensor::prime(U)*d*itensor::dag(U);
        }

        itensor::ITensor ZigZag_Bond::Swap(size_t site_idx) {
                if (swap_gates_.find(site_idx) == swap_gates_.end()) {
                        auto swap_gate = itensor::ITensor(itensor::dag(sites_(site_idx)), itensor::dag(sites_(site_idx+1)),
                                                          itensor::prime(sites_(site_idx)), itensor::prime(sites_(site_idx+1)));
                        size_t m1 = itensor::dim(sites_(site_idx)), m2 = itensor::dim(sites_(site_idx+1));
                        for (size_t i = 1; i <= m1; i++) {
                                for (size_t j = 1; j <= m2; j++) {
                                        swap_gate.set(itensor::dag(sites_(site_idx))=i, itensor::dag(sites_(site_idx+1))=j,
                                                      itensor::prime(sites_(site_idx))=j, itensor::prime(sites_(site_idx+1))=i, 1.0);
                                }
                        }
                        swap_gates_[site_idx] = swap_gate;
                        return swap_gate;
                }

                return swap_gates_[site_idx];
        }
} // namespace ZigZag_TEBD
#endif //UUID_95124DA6_52D9_4AE3_8F5D_2ACA5ECDC5C5
