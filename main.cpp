#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <functional>
#include <fmt/core.h>

namespace ch = std::chrono;

#define NUMERO_ITERACIONES 1024

typedef std::function<float(float, float)> binary_op;

float sum_reduction_serial(const float *input, const int n, binary_op op) {
    float suma = 0.0f;
    for (int i = 0; i < n; ++i)
        suma = op(suma, input[i]);
    return suma;
}

float reduccion_paralela(const float *input, const int n, binary_op op) {
    int num_threads;
    long block_size;
    float *resultados_parciales;

#pragma omp parallel default(none) shared(num_threads, block_size, resultados_parciales, op, n, input)
    {
#pragma omp master
        {
            num_threads = omp_get_num_threads();
            block_size = n / num_threads;
            resultados_parciales = new float[num_threads];
            fmt::println("Numero hilos {}, block size {}", num_threads, block_size);
        }
#pragma omp barrier

        int thread_id = omp_get_thread_num();
        long start = thread_id * block_size;
        long end = (thread_id + 1) * block_size;

        if (thread_id == num_threads - 1) {
            end = n;
        }

        float suma_parcial = 0.0f;
        for (int i = start; i < end; i++) {
            suma_parcial = op(suma_parcial, input[i]);
        }

        resultados_parciales[thread_id] = suma_parcial;

#pragma omp barrier
        for (int i = num_threads / 2; i > 0; i /= 2) {
#pragma omp barrier
            if (thread_id < i) {
                resultados_parciales[thread_id] = op(resultados_parciales[thread_id], resultados_parciales[thread_id + i]);
            }
        }

#pragma omp master
        {
            resultados_parciales[0] = op(resultados_parciales[0], 0.0f);
        }
    }

    float suma_total = resultados_parciales[0];
    delete[] resultados_parciales;
    return suma_total;
}

int main() {
    std::vector<float> datos(NUMERO_ITERACIONES);

    for (int i = 0; i < NUMERO_ITERACIONES; i++) {
        datos[i] = i + 1;
    }

    auto op_add = [](auto v1, auto v2) {
        return v1 + v2;
    };

    fmt::println("////////////////////////////");
    fmt::println(" ");

    auto suma_serial = sum_reduction_serial(datos.data(), datos.size(), op_add);
    fmt::println("Reduccion serial: {}", suma_serial);

    fmt::println(" ");
    fmt::println("////////////////////////////");
    fmt::println(" ");

    auto suma_paralela = reduccion_paralela(datos.data(), datos.size(), op_add);
    fmt::println("Reduccion paralela: {}", suma_paralela);

    fmt::println(" ");
    fmt::println("////////////////////////////");
    fmt::println(" ");

    return 0;
}
