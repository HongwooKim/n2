// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(__GNUC__)
  #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
  #define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#include <iostream>
#include <utility>  // For ::std::pair.
#include <vector>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <xmmintrin.h>


#include <numeric>


namespace n2 {

using std::endl;
using std::fstream;
using std::max;
using std::min;
using std::mutex;
using std::ofstream;
using std::ifstream;
using std::pair;
using std::priority_queue;
using std::setprecision;
using std::string;
using std::stof;
using std::stoi;
using std::to_string;
using std::unique_lock;
using std::unordered_set;
using std::vector;

class BaseDistance {
    public:
    BaseDistance() {}
    virtual const vector<float> initQvec(vector<float> qvec) const = 0;


    ::std::pair<int, float> TraverseLevels(int level,char* model_level0_, long long memory_per_node_level0_, long long memory_per_node_higher_level_,char* model_higher_level_,long long memory_per_link_level0_,
     float cur_dist, int cur_node_id,const float* qraw,size_t data_dim_,bool ensure_k_,vector<pair<int, float> > path, float  *  __restrict TmpRes) ;
    virtual ~BaseDistance() = 0;
    virtual float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const = 0;

};

class L2Distance : public BaseDistance {
   public:
   L2Distance() {}
   ~L2Distance() override {}
   const vector<float> initQvec(vector<float> qvec) const override;
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

class AngularDistance : public BaseDistance {
   public:
   AngularDistance() {}
   ~AngularDistance() override {}
   const vector<float> initQvec(vector<float> qvec) const override;
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

} // namespace n2
