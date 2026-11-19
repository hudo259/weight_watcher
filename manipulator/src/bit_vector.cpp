#include "bit_vector.hpp"

#include <cstring>


void BitVector::resize(size_t num_bits) {
    num_bits_ = num_bits;
    data_.resize((num_bits + 31) / 32, 0);
}

void BitVector::clear() {
    std::memset(data_.data(), 0, data_.size() * sizeof(uint32_t));
}

bool BitVector::get(size_t idx) const {
    if (idx >= num_bits_) return false;

    uint32_t block = data_[idx / 32];
    //moves bit at position idx in the block right to position 0 and checks if it is set or not
    return (block >> (idx % 32)) & 1u;
}

void BitVector::set(size_t idx, bool value) {
    if (idx >= num_bits_) return;

    if (value) {
        //create a mask by shifting the 1 from position 0 (1u) to position idx
        //then we just bitwise OR the mask (bit at idx) with what is in the word at block [idx / 32]
        //=> either the bit at idx is set then it remains set or it is not set, then it gets set
        data_[idx / 32] |= (1u << (idx % 32));
    } else {
        //create a mask by shifting the 1 from position 0 (1u) to position idx and negate the whole word, so the bits are negated
        //then we just bitwise AND with the word at block [idx / 32]
        //=> we keep bits that are set to 1 in the word that are not at position idx. The bit at idx is set to 0 if it was 1 or it remains 0
        data_[idx / 32] &= ~(1u << (idx % 32));
    }
}

void BitVector::toggle(size_t idx) {
    if (idx >= num_bits_) return;

    //create a mask by shifting the 1 from position 0 (1u) to position idx
    //bitwise XOR with word at block idx/32. => Flips the target bit at position idx and keeps the rest unchanged
    data_[idx / 32] ^= (1u << (idx % 32));
}

bool BitVector::operator[](size_t idx) const { 
    return get(idx); 
}

size_t BitVector::size() const { 
    return num_bits_; 
}

size_t BitVector::size_in_bytes() const { 
    return data_.size() * sizeof(uint32_t); 
}

const uint32_t* BitVector::data() const { 
    return data_.data(); 
}

uint32_t* BitVector::data() { 
    return data_.data(); 
}

size_t BitVector::count() const {
    size_t total = 0;

    //we use compiler-specific functions to count the number of 1s per word. It's more efficient than iterating over each bit manually
    for (uint32_t word : data_) {
        #ifdef _MSC_VER
            total += __popcnt(word);
        #else
            total += __builtin_popcount(word);
        #endif
    }
    return total;
}