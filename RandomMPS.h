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

#ifndef UUID_44723946_6E0D_4CAD_94D7_D0472947F58D
#define UUID_44723946_6E0D_4CAD_94D7_D0472947F58D

#include <itensor/all_mps.h>
#include "RandomPhaseState.h"
#include <chrono>
#include <cmath>
#include <string>
#include <random>
#include <toml.hpp>
#include <utility>
#include <json.hpp>
#include <vector>


namespace randomMPS {
        class Sampler {
                private:
                        uint_fast64_t seed_;
                        double dBeta_;
                        int NBeta_, ObserveInterval_, n_uni_, count_;
                        nlohmann::json output_;
                        std::mt19937_64 engine_;
                        itensor::Args tevol_args_;
                        std::vector<double> beta_;
                        std::string filename_;
                        std::vector<std::vector<itensor::QN>> PossibleQNs_;
                        itensor::SiteSet sites_;
                        std::vector<std::pair<int, itensor::ITensor>> gates_, uni_gates_;

                        void initialize();

                public:
                        Sampler(const itensor::SiteSet &sites, uint_fast64_t seed);
                        Sampler(const itensor::SiteSet &sites);
                        void set_target(itensor::QN target) {
                                PossibleQNs_ = RandomPhaseState::GeneratePossibleQNs(sites_, target);
                        }
                        void set_gates(const std::vector<std::pair<int, itensor::ITensor>> &gates) { gates_ = gates; }
                        void set_unitary(const std::vector<std::pair<int, itensor::ITensor>> &gates) { uni_gates_ = gates; }
                        template <typename T>
                        void run(T& observer);
        };

        Sampler::Sampler(const itensor::SiteSet &sites, uint_fast64_t seed) : seed_(seed), sites_(sites) {
                initialize();
        }

        Sampler::Sampler(const itensor::SiteSet &sites) : sites_(sites) {
                std::random_device seed_gen;
                uint_fast64_t seed1 = seed_gen();
                uint_fast64_t seed2 = seed_gen();
                seed_ = (seed1 << 32) + seed2;

                initialize();
        }

        void Sampler::initialize() {
                engine_.seed(seed_);

                const auto toml = toml::parse("setting.toml");
                dBeta_ = toml::find<double>(toml, "tDMRG", "dBeta");
                NBeta_ = toml::find<int>(toml, "tDMRG", "NBeta");
                ObserveInterval_ = toml::find<int>(toml, "Sampling", "ObserveInterval");

                if (toml.contains("UnitaryTransformation")) {
                        n_uni_ = toml::find<int>(toml, "UnitaryTransformation", "Steps");
                } else {
                        n_uni_ = 0;
                }

                int MaxM = toml::find<int>(toml, "MPS", "MaxM");
                double tol = toml::find<double>(toml, "MPS", "tol");

                tevol_args_ = itensor::Args(
                                "MaxDim", MaxM,
                                "Cutoff", tol
                                );

                output_["seed"] = seed_;

                beta_.reserve(NBeta_ / ObserveInterval_ + 1);
                for (int i = 0; i < NBeta_; i++) {
                        if (i % ObserveInterval_ == 0) {
                                beta_.push_back(i*dBeta_);
                        }
                }
                beta_.push_back(NBeta_*dBeta_);
                output_["beta"] = beta_;

                output_["LowestEnergy"] = nullptr;

                std::string file_pre("sample_"), file_suf(".json"), seed_str;
                seed_str = std::to_string(seed_);
                filename_ = file_pre + seed_str + file_suf;

                count_ = 0;
        }

        template<typename T>
        void Sampler::run(T& observer) {

                        nlohmann::json sample;
                        auto start = std::chrono::system_clock::now();
                        itensor::MPS psi;
                        if (PossibleQNs_.size() > 0) {
                                psi = RandomPhaseState::RandomPhaseState(sites_, PossibleQNs_, engine_);
                        } else {
                                psi = RandomPhaseState::RandomPhaseState(sites_, engine_);
                        }

                        double nrm = psi.normalize();
                        for (int i = 0; i < n_uni_; i++) {
                                for (auto&& x : uni_gates_) {
                                        psi.position(x.first);
                                        itensor::applyGate(x.second, psi, tevol_args_);
                                        nrm *= psi.normalize();
                                }
                        }
                        double ene_for_norm = 0.0;
                        double norm_factor = 1.0;

                        for (int i = 0; i < NBeta_; i++) {
                                if (i % ObserveInterval_ == 0) {
                                        observer(psi, sample);
                                        sample["Norm"].push_back(nrm);
                                        sample["BondDim"].push_back(itensor::maxLinkDim(psi));
                                        double ene_present = sample["Energy"].back();

                                        if (i*0.5*dBeta_*std::abs(ene_present - ene_for_norm) > 1.0) {
                                                double ene_diff = ene_present - ene_for_norm;
                                                for (size_t j = 1; j < sample["Norm"].size(); j++) {
                                                        double val = sample["Norm"].at(j);
                                                        sample["Norm"].at(j) = val*std::exp(0.5*static_cast<double>(output_["beta"].at(j))*ene_diff);
                                                }
                                                nrm *= std::exp(i*0.5*dBeta_*ene_diff);
                                                norm_factor *= std::exp(0.5*dBeta_*ene_diff);
                                                ene_for_norm = ene_present;
                                        }
                                }

                                for (auto&& x : gates_) {
                                        psi.position(x.first);
                                        itensor::applyGate(x.second, psi, tevol_args_);
                                        nrm *= psi.normalize();
                                }
                                nrm *= norm_factor;

                        }
                        observer(psi, sample);
                        sample["Norm"].push_back(nrm);
                        sample["BondDim"].push_back(itensor::maxLinkDim(psi));
                        double ene_present = sample["Energy"].back();
                        double ene_diff = ene_present - ene_for_norm;
                        for (size_t j = 1; j < sample["Norm"].size(); j++) {
                                double val = sample["Norm"].at(j);
                                sample["Norm"].at(j) = val*std::exp(0.5*static_cast<double>(output_["beta"].at(j))*ene_diff);
                        }

                        if (output_["LowestEnergy"].is_null() or output_["LowestEnergy"] > ene_present) {
                                output_["LowestEnergy"] = ene_present;
                        }

                        output_["Samples"].push_back(sample);
                        auto end = std::chrono::system_clock::now();
                        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                        count_++;
                        std::cout << "Sample " << count_ << ", Elapsed time:" << elapsed / 1000 << "s, Norm:" << sample["Norm"].back() << std::endl;
                        output_["ElapsedTime"].push_back(elapsed/1000);
                        std::ofstream out_file(filename_);
                        out_file << output_ << std::endl;
                }
} // namespace randomMPS
#endif //UUID_44723946_6E0D_4CAD_94D7_D0472947F58D
