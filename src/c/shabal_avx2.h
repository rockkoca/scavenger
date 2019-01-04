#pragma once

#include <stdint.h>
#include <stdlib.h>
//#include "mshabal_256.h"

void init_shabal_avx2();

void find_best_deadline_avx2(char* scoops, uint64_t nonce_count, char* gensig,
                             uint64_t* best_deadline, uint64_t* best_offset);
