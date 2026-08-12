

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_X       100
#define TAG_CONTROL 101
#define TAG_RESULT  200

typedef unsigned long long ull;

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  mpiexec -n <P> %s <x> <samples> [seed] [--csv]\n\n"
        "Arguments:\n"
        "  x        Upper limit of integral: u(x) = integral_0^x sin(t) dt\n"
        "  samples  Total number of Monte Carlo samples, positive integer\n"
        "  seed     Optional unsigned integer seed. Default: 123456789\n"
        "  --csv    Optional: print one CSV row instead of human-readable output\n\n"
        "Examples:\n"
        "  mpiexec -n 4 %s 1.5707963267948966 10000000 12345\n"
        "  mpiexec -n 4 %s 1.5707963267948966 10000000 12345 --csv\n",
        prog, prog, prog);
}

static int parse_double_arg(const char *s, double *value) {
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) {
        return 0;
    }
    *value = v;
    return 1;
}

static int parse_ull_arg(const char *s, ull *value) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    *value = (ull)v;
    return 1;
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double rng_uniform_01(uint64_t *state) {
    return (double)(splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static ull seed_for_rank(ull base_seed, int rank) {
    uint64_t s = (uint64_t)base_seed;
    s ^= UINT64_C(0x9E3779B97F4A7C15) * (uint64_t)(rank + 1);
    s ^= UINT64_C(0xBF58476D1CE4E5B9) + ((uint64_t)rank << 32);
    return (ull)s;
}

static double local_monte_carlo_sum(double x, ull local_n, ull seed, double *compute_time) {
    uint64_t state = (uint64_t)seed;
    double sum = 0.0;
    double compensation = 0.0;

    double start = MPI_Wtime();

    for (ull i = 0; i < local_n; i++) {
        double r = rng_uniform_01(&state);
        double t = x * r;
        double y = sin(t) - compensation;
        double tmp = sum + y;
        compensation = (tmp - sum) - y;
        sum = tmp;
    }

    double finish = MPI_Wtime();
    if (compute_time != NULL) {
        *compute_time = finish - start;
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int comm_sz = 0;
    int my_rank = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double x = 0.0;
    ull total_samples = 0ULL;
    ull base_seed = 123456789ULL;
    int csv_mode = 0;

    if (argc < 3) {
        if (my_rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        if (my_rank == 0) print_usage(argv[0]);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (!parse_double_arg(argv[1], &x) || !parse_ull_arg(argv[2], &total_samples) || total_samples == 0ULL) {
        if (my_rank == 0) {
            fprintf(stderr, "Error: invalid <x> or <samples>.\n\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int seed_seen = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv_mode = 1;
        } else if (!seed_seen) {
            if (!parse_ull_arg(argv[i], &base_seed)) {
                if (my_rank == 0) {
                    fprintf(stderr, "Error: invalid seed: %s\n\n", argv[i]);
                    print_usage(argv[0]);
                }
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            seed_seen = 1;
        } else {
            if (my_rank == 0) {
                fprintf(stderr, "Error: unknown extra argument: %s\n\n", argv[i]);
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    double global_sum = 0.0;
    double max_worker_compute_time = 0.0;
    int workers = comm_sz > 1 ? comm_sz - 1 : 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double local_start = MPI_Wtime();

    if (comm_sz == 1) {
        double compute_time = 0.0;
        global_sum = local_monte_carlo_sum(x, total_samples, seed_for_rank(base_seed, 0), &compute_time);
        max_worker_compute_time = compute_time;
    } else if (my_rank == 0) {
        ull base = total_samples / (ull)workers;
        ull rem = total_samples % (ull)workers;

        for (int dest = 1; dest < comm_sz; dest++) {
            ull local_n = base + ((ull)dest <= rem ? 1ULL : 0ULL);
            ull control[2];
            control[0] = local_n;
            control[1] = seed_for_rank(base_seed, dest);

            MPI_Send(&x, 1, MPI_DOUBLE, dest, TAG_X, MPI_COMM_WORLD);
            MPI_Send(control, 2, MPI_UNSIGNED_LONG_LONG, dest, TAG_CONTROL, MPI_COMM_WORLD);
        }

        for (int src = 1; src < comm_sz; src++) {
            double result[2] = {0.0, 0.0};
            MPI_Recv(result, 2, MPI_DOUBLE, src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_sum += result[0];
            if (result[1] > max_worker_compute_time) {
                max_worker_compute_time = result[1];
            }
        }
    } else {
        double local_x = 0.0;
        ull control[2] = {0ULL, 0ULL};
        MPI_Recv(&local_x, 1, MPI_DOUBLE, 0, TAG_X, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(control, 2, MPI_UNSIGNED_LONG_LONG, 0, TAG_CONTROL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        ull local_n = control[0];
        ull local_seed = control[1];
        double compute_time = 0.0;
        double local_sum = local_monte_carlo_sum(local_x, local_n, local_seed, &compute_time);

        double result[2];
        result[0] = local_sum;
        result[1] = compute_time;
        MPI_Send(result, 2, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }

    double local_finish = MPI_Wtime();
    double local_elapsed = local_finish - local_start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double estimate = x * (global_sum / (double)total_samples);
        double exact = 1.0 - cos(x);
        double abs_error = fabs(estimate - exact);
        double overhead_estimate = elapsed - max_worker_compute_time;
        if (overhead_estimate < 0.0) overhead_estimate = 0.0;

        if (csv_mode) {
            printf("%d,%d,%.17g,%llu,%llu,%.17g,%.17g,%.17g,%.9f,%.9f,%.9f\n",
                   comm_sz, workers, x, total_samples, base_seed,
                   estimate, exact, abs_error,
                   elapsed, max_worker_compute_time, overhead_estimate);
        } else {
            printf("Monte Carlo MPI approximation\n");
            printf("Problem: u(x) = integral_0^x sin(t) dt\n");
            printf("MPI processes              : %d\n", comm_sz);
            printf("Workers                    : %d\n", workers);
            printf("x                          : %.17g\n", x);
            printf("Total samples              : %llu\n", total_samples);
            printf("Seed                       : %llu\n", base_seed);
            printf("Estimate                   : %.17g\n", estimate);
            printf("Exact 1 - cos(x)           : %.17g\n", exact);
            printf("Absolute error             : %.17g\n", abs_error);
            printf("Elapsed time, max over MPI : %.9f seconds\n", elapsed);
            printf("Max worker compute time    : %.9f seconds\n", max_worker_compute_time);
            printf("Overhead estimate          : %.9f seconds\n", overhead_estimate);
            if (comm_sz == 1) {
                printf("Note                       : P=1 uses serial fallback, not Master-Worker.\n");
            }
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
