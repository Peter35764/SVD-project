// Файл: src/test_common.cpp

#include <mutex> // Для std::mutex

// Определение глобального мьютекса.
// Он объявлен как 'extern std::mutex cout_mutex;' в test_common.h
std::mutex cout_mutex;