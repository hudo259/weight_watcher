#pragma once

#include <vector>
#include <cstdint>

// Custom compact bitmask for selection: 1 bit per point. std::vector<bool> uses 1 byte per bool because the C++ committee sucks ass. 
// And std::bitset is not dynamic and therefore useless.  
// Efficient operations using popcount and bit scanning. Uploads directly to SSBO for GPU access.
class BitVector {
public:
    explicit BitVector(size_t num_bits = 0) 
        : num_bits_(num_bits)
        , data_((num_bits + 31) / 32, 0) 
    {}
    
    void resize(size_t num_bits);
    void clear();
    bool get(size_t idx) const;
    void set(size_t idx, bool value);
    void toggle(size_t idx);
    bool operator[](size_t idx) const;
    size_t size() const;
    size_t size_in_bytes() const;
    const uint32_t* data() const;
    uint32_t* data();
    size_t count() const;
    
    // Applying func over each bit and iterating over set bits efficiently
    template<typename Func>
    void for_each_set(Func&& func) const {
        for (size_t word_idx = 0; word_idx < data_.size(); word_idx++) {
            uint32_t word = data_[word_idx];

            //if at least one bit is set, then word is > 0
            while (word) {
                //We count trailing zeros with these functions => bit_idx is the lowest set bit
                #ifdef _MSC_VER
                    unsigned long bit_idx;
                    _BitScanForward(&bit_idx, word);
                #else
                    uint32_t bit_idx = __builtin_ctz(word);
                #endif
                
                size_t global_idx = word_idx * 32 + bit_idx;
                if (global_idx < num_bits_) {
                    func(global_idx);
                }
                word &= word - 1;  // Clear lowest set bit
            }
        }
    }

private:
    size_t num_bits_;
    std::vector<uint32_t> data_;
};