#include <mpi.h>
#include <locale.h>
#include <iostream>
#include <vector>
#include <fstream>

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "ukr");
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<int> local_data;
    std::vector<int> full_data;
    std::vector<int> sendcounts, displs;
    long long total_elements = 0;

    if (world_rank == 0) {
        std::ifstream in("D:\\Projects_OOP\\generate_array\\array_1B.bin", std::ios::binary);
        if (!in) {
            std::cerr << "Не вдалося вiдкрити файл!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        in.read(reinterpret_cast<char*>(&total_elements), sizeof(total_elements));
        full_data.resize(total_elements);
        in.read(reinterpret_cast<char*>(full_data.data()), total_elements * sizeof(int));
        in.close();

        // Підготовка sendcounts і displs
        sendcounts.resize(world_size);
        displs.resize(world_size);
        long long base = total_elements / world_size;
        long long remainder = total_elements % world_size;
        long long offset = 0;

        for (int i = 0; i < world_size; ++i) {
            sendcounts[i] = static_cast<int>(base + (i < remainder ? 1 : 0));
            displs[i] = static_cast<int>(offset);
            offset += sendcounts[i];
        }
    }

    // Надсилаємо розмір масиву
    MPI_Bcast(&total_elements, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    // Розсилка sendcounts і displs
    if (world_rank != 0) {
        sendcounts.resize(world_size);
        displs.resize(world_size);
    }

    MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    local_data.resize(sendcounts[world_rank]);

    // Розподілення частин
    MPI_Scatterv(full_data.data(), sendcounts.data(), displs.data(), MPI_INT,
        local_data.data(), sendcounts[world_rank], MPI_INT,
        0, MPI_COMM_WORLD);

    double start = MPI_Wtime();

    // Локальна сума
    long long local_sum = 0;
    for (int val : local_data) {
        local_sum += val;
    }

    // Збір глобальної суми
    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "Sum: " << global_sum << std::endl;
        std::cout << "Time: " << (end - start) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
