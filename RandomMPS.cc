// Licensed under the MIT License <http://opensource.org/MIT>
//
// Copyright (c) 2020 Shimpei Goto
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnishedto do so, subject to the following conditions:
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

#include <itensor/all_mps.h>
#include "RandomMPS.h"
#include "ZigZag_bond.h"
#include "XXZ_bond.h"
#include <algorithm>
#include <stdexcept>
#include <toml.hpp>
#include <json.hpp>

namespace  {
        class Observer {
                private:
                        itensor::MPO H_;

                public:
                        Observer(itensor::MPO &H) : H_(H) {};
                        void operator()(itensor::MPS &psi, nlohmann::json &sample);
        };

        void Observer::operator()(itensor::MPS &psi, nlohmann::json &sample) {
                sample["SquaredEnergy"].push_back(itensor::innerC(itensor::prime(psi, 2), itensor::prime(H_, 1), H_, psi).real());
                sample["Energy"].push_back(itensor::innerC(psi, H_, psi).real());
        }
} // namespace

int main() {
        const auto toml = toml::parse("setting.toml");
        double J = toml::find<double>(toml, "System", "J");
        double J2 = toml::find<double>(toml, "System", "J2");
        int Ns = toml::find<int>(toml, "System", "Lattice");
        int Sz = toml::find<int>(toml, "System", "Sz");
        bool is_abelian = toml::find<bool>(toml, "System", "AbelianSymmetry");
        double hz;
        if (!is_abelian) {
                hz = toml::find<double>(toml, "System", "MagneticField");
        }

        double dBeta = toml::find<double>(toml, "tDMRG", "dBeta");

        int NSample = toml::find<int>(toml, "Sampling", "Sample");

        auto sites = itensor::SpinHalf(Ns, {"ConserveQNs", is_abelian});

        // Define Hamiltonian
        auto ampo_H = itensor::AutoMPO(sites);
        for (int i = 1; i <= Ns; i++) {
                if (!is_abelian) {
                        ampo_H += hz, "Sz", i;
                }
                if (i+1 <= Ns) {
                        ampo_H += 0.5*J, "S+", i, "S-", i+1;
                        ampo_H += 0.5*J, "S-", i, "S+", i+1;
                        ampo_H += J2, "Sz", i, "Sz", i+1;
                }

                if (i+2 <= Ns and std::abs(J2) >= 1e-8) {
                        ampo_H += 0.5*J2, "S+", i, "S-", i+2;
                        ampo_H += 0.5*J2, "S-", i, "S+", i+2;
                        ampo_H += J2, "Sz", i, "Sz", i+2;
                }
        }

        auto H = itensor::toMPO(ampo_H);
        auto obs = Observer(H);

        int Nup = (Ns + Sz) / 2, Ndn = (Ns - Sz) / 2;
        if (Nup + Ndn != Ns or Nup < 0 or Ndn < 0 or Nup > Ns or Ndn > Ns) {
                throw std::runtime_error("Selected Sz sector cannot be specified in this system");
        }

        // Setup Trotter gates
        ZigZag_Trotter::ZigZag_Bond sys(Ns, J, J2, sites);
        std::vector<std::pair<int, itensor::ITensor>> gates;
        int n_gates = 2*(Ns-1);
        if (std::abs(J2) >= 1e-8) {
                n_gates += 2*3*(Ns-2);
        }
        gates.reserve(n_gates);
        for (int i = 1; i <= Ns-1; i+=2) {
                gates.emplace_back(i, sys.BondTerm(i, i+1, -0.25*dBeta, i));
        }
        for (int i = 2; i <= Ns-1; i+=2) {
                gates.emplace_back(i, sys.BondTerm(i, i+1, -0.25*dBeta, i));
        }

        if (std::abs(J2) >= 1e-8) {
                for (int i = 1; i <= Ns-2; i++) {
                        gates.emplace_back(i+1, sys.Swap(i+1));
                        gates.emplace_back(i, sys.BondTerm(i, i+2, -0.25*dBeta, i));
                        gates.emplace_back(i+1, sys.Swap(i+1));
                }
        }
        auto reversed = gates;
        std::reverse(reversed.begin(), reversed.end());
        for (auto&& x : reversed) {
                gates.push_back(x);
        }

        randomMPS::Sampler Sampler(sites);
        if (is_abelian) {
                itensor::QN target({"Sz", Sz});
                Sampler.set_target(target);
        }

        Sampler.set_gates(gates);
        if (toml.contains("UnitaryTransformation")){
                double tau_uni = toml::find<double>(toml, "UnitaryTransformation", "tau");
                double Jz_uni = toml::find<double>(toml, "UnitaryTransformation", "Jz");

                XXZ_Trotter::XXZ_Bond sys_uni(Ns, J, Jz_uni, sites);
                std::vector<std::pair<int, itensor::ITensor>> unitary;
                unitary.reserve(Ns-1);
                for (int i = 1; i <= Ns-1; i+=2) {
                        unitary.emplace_back(i, sys_uni.BondTerm(i, i+1, {0.0, -tau_uni}, i));
                }
                for (int i = 2; i <= Ns-1; i+=2) {
                        unitary.emplace_back(i, sys_uni.BondTerm(i, i+1, {0.0, -tau_uni}, i));
                }
                Sampler.set_unitary(unitary);
        }

        for (int i = 0; i < NSample; i++) {
                Sampler.run(obs);
        }

        return 0;
}
