#pragma once


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__                               \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__                               \
    }                                           \

