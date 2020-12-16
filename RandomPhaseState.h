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

#ifndef UUID_DFB5FFA9_FC7D_4EE2_8096_AE8280CCE961
#define UUID_DFB5FFA9_FC7D_4EE2_8096_AE8280CCE961
#include <itensor/all_mps.h>
#include <algorithm>
#include <complex>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace RandomPhaseState {
        std::vector<std::vector<itensor::QN>> GeneratePossibleQNs(const itensor::SiteSet &sites, const itensor::QN &target)
        {
                if (!itensor::hasQNs(sites))
                {
                        throw std::runtime_error("Sites should have QNs!");
                }
                int N = itensor::length(sites);

                // From Left to Right
                std::vector<std::vector<itensor::QN>> FromLeftToRight;
                FromLeftToRight.reserve(N+1);
                FromLeftToRight.push_back({target});
                for (int i = 1; i < N; i++)
                {
                        std::vector<itensor::QN> qn_vec;
                        int nblock = itensor::nblock(sites(i));
                        qn_vec.reserve(nblock*FromLeftToRight[i-1].size());
                        for (int j = 1; j <= nblock; j++)
                        {
                                auto qn_right = itensor::qn(sites(i), j);
                                for (auto&& qn_left : FromLeftToRight[i-1])
                                {
                                        auto qn_tmp = qn_left - qn_right;
                                        if (std::find(qn_vec.begin(), qn_vec.end(), qn_tmp) == qn_vec.end())
                                        {
                                                qn_vec.push_back(qn_tmp);
                                        }
                                }
                        }
                        FromLeftToRight.push_back(qn_vec);
                }
                FromLeftToRight.push_back({itensor::QN()});

                // From Right to Left
                std::vector<std::vector<itensor::QN>> FromRightToLeft;
                FromRightToLeft.reserve(N+1);
                FromRightToLeft.push_back({itensor::QN()});
                for (int i = 1; i < N; i++)
                {
                        std::vector<itensor::QN> qn_vec;
                        int nblock = itensor::nblock(sites(N+1-i));
                        qn_vec.reserve(nblock*FromRightToLeft[i-1].size());
                        for (int j = 1; j <= nblock; j++)
                        {
                                auto qn_left = itensor::qn(sites(N+1-i), j);
                                for (auto&& qn_right : FromRightToLeft[i-1])
                                {
                                        auto qn_tmp = qn_right + qn_left;
                                        if (std::find(qn_vec.begin(), qn_vec.end(), qn_tmp) == qn_vec.end())
                                        {
                                                qn_vec.push_back(qn_tmp);
                                        }
                                }
                        }
                        FromRightToLeft.push_back(qn_vec);
                }
                FromRightToLeft.push_back({target});

                // Compare
                std::vector<std::vector<itensor::QN>> PossibleQNs;
                PossibleQNs.reserve(N+1);

                for (int i = 0; i <= N; i++)
                {
                        std::vector<itensor::QN> qn_vec;
                        qn_vec.reserve(std::min(FromLeftToRight[i].size(), FromRightToLeft[N-i].size()));
                        for (auto&& FromLeft : FromLeftToRight[i])
                        {
                                if (std::find(FromRightToLeft[N-i].begin(), FromRightToLeft[N-i].end(), FromLeft) != FromRightToLeft[N-i].end())
                                {
                                        qn_vec.push_back(FromLeft);
                                }
                        }
                        PossibleQNs.push_back(qn_vec);
                }

                return PossibleQNs;
        }

        itensor::MPS RandomPhaseState(const itensor::SiteSet &sites, const itensor::QN &target, std::mt19937_64 &engine)
        {
                auto PossibleQNs = GeneratePossibleQNs(sites, target);
                int N = itensor::length(sites);
                itensor::MPS psi(sites);
                std::vector<itensor::Index> links(N+1);
                for (int l = 0; l <= N; l++)
                {
                        std::vector<std::pair<itensor::QN, long>> qnstorage;
                        auto ts = itensor::format("Link,l=%d", l);
                        for (auto&& qn : PossibleQNs[l])
                        {
                                qnstorage.emplace_back(qn, 1);
                        }
                        links[l] = itensor::Index(std::move(qnstorage), itensor::Out, ts);
                }

                std::uniform_real_distribution<> dist(0.0, 4.0*std::acos(0.0));

                for (int n = 1; n <= N; n++)
                {
                        auto & A = psi.ref(n);
                        auto row = itensor::dag(links.at(n-1));
                        auto col = links.at(n);

                        A = itensor::ITensor(sites(n), row, col);
                        for (int d = 1; d <= itensor::dim(sites(n)); d++)
                        {
                                for (int l = 1; l <= itensor::dim(row); l++)
                                {
                                        for (int k = 1; k <= itensor::dim(col); k++)
                                        {
                                                if (row.qn(l) - sites(n).qn(d) == col.qn(k))
                                                {
                                                        double theta = dist(engine);
                                                        A.set(sites(n)(d), row(l), col(k), std::complex<double>(std::cos(theta), std::sin(theta)));
                                                }
                                        }
                                }
                        }
                }
                psi.ref(1) *= itensor::setElt(links.at(0)(1));
                psi.ref(N) *= itensor::setElt(itensor::dag(links.at(N)(1)));

                psi.position(1);
                std::cout << "overlap:" << itensor::innerC(psi, psi) << std::endl;
                psi.normalize();

                return psi;
        }

        itensor::MPS RandomPhaseState(const itensor::SiteSet &sites, const std::vector<std::vector<itensor::QN>> &PossibleQNs, std::mt19937_64 &engine)
        {
                int N = itensor::length(sites);
                itensor::MPS psi(sites);
                std::vector<itensor::Index> links(N+1);
                for (int l = 0; l <= N; l++)
                {
                        std::vector<std::pair<itensor::QN, long>> qnstorage;
                        auto ts = itensor::format("Link,l=%d", l);
                        for (auto&& qn : PossibleQNs[l])
                        {
                                qnstorage.emplace_back(qn, 1);
                        }
                        links[l] = itensor::Index(std::move(qnstorage), itensor::Out, ts);
                }

                std::uniform_real_distribution<> dist(0.0, 4.0*std::acos(0.0));

                for (int n = 1; n <= N; n++)
                {
                        auto & A = psi.ref(n);
                        auto row = itensor::dag(links.at(n-1));
                        auto col = links.at(n);

                        A = itensor::ITensor(sites(n), row, col);
                        for (int d = 1; d <= itensor::dim(sites(n)); d++)
                        {
                                for (int l = 1; l <= itensor::dim(row); l++)
                                {
                                        for (int k = 1; k <= itensor::dim(col); k++)
                                        {
                                                if (row.qn(l) - sites(n).qn(d) == col.qn(k))
                                                {
                                                        double theta = dist(engine);
                                                        A.set(sites(n)(d), row(l), col(k), std::complex<double>(std::cos(theta), std::sin(theta)));
                                                }
                                        }
                                }
                        }
                }
                psi.ref(1) *= itensor::setElt(links.at(0)(1));
                psi.ref(N) *= itensor::setElt(itensor::dag(links.at(N)(1)));

                psi.position(1);

                return psi;
        }

        itensor::MPS RandomPhaseState(const itensor::SiteSet &sites, std::mt19937_64 &engine)
        {
                int N = itensor::length(sites);
                if (itensor::hasQNs(sites(1))) {
                        throw std::runtime_error("Sites should not have QNs!");
                }
                itensor::MPS psi(sites);
                std::vector<itensor::Index> links(N+1);
                for (int l = 0; l <= N; l++) {
                        auto ts = itensor::format("Link,l=%d", l);
                        links.at(l) = itensor::Index(1, ts);
                }

                std::uniform_real_distribution<> dist(0.0, 4.0*std::acos(0.0));
                for (int n = 1; n <= N; n++) {
                        auto &A = psi.ref(n);
                        auto row = links.at(n-1);
                        auto col = links.at(n);

                        A = itensor::ITensor(sites(n), row, col);
                        for (int d = 1; d <= itensor::dim(sites(n)); d++) {
                                double theta = dist(engine);
                                A.set(sites(n)(d), row(1), col(1), std::complex<double>(std::cos(theta), std::sin(theta)));
                        }
                }

                psi.ref(1) *= itensor::setElt(links.at(0)(1));
                psi.ref(N) *= itensor::setElt(links.at(N)(1));

                psi.position(1);

                return psi;
        }
} // namespace RandomPhaseState
#endif //UUID_DFB5FFA9_FC7D_4EE2_8096_AE8280CCE961
