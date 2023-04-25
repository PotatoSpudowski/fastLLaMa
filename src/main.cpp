#include "bridge.hpp"
#include <thread>
#include <chrono>
#include <numeric>
#include "concurrency/utils.hpp"

using namespace std::chrono_literals;
using namespace fastllama;

int main() {
    auto pool = ThreadPool{8};

    pool.start();

    auto natural_number = std::vector(100, 0);
    std::iota(natural_number.begin(), natural_number.end(), 0);
    
    // parallel::for_(pool, parallel::Range{0, 100, 10}, [](parallel::Block block) {
    //     std::stringstream ss;
    //     ss <<"Hello from worker thread "<<std::this_thread::get_id()<<" with block "<<block.start<<" "<<block.end<<"\n";
    //     std::cout<<ss.str();
    // });

    parallel::transform(pool, natural_number, [](auto a) {
        return a * 2;
    });

    auto sum = 0ul;
    for(auto i = 0; i < natural_number.size(); ++i) {
        sum += natural_number[i];
    }

    auto result = parallel::reduce(pool, natural_number, 0, [](auto a, auto b) {
        return a + b;
    });


    std::cout<<"Result: "<<result << " == " << sum <<"\n";

    return 0;
}