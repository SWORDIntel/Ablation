/**
 * NOT_STISLA Real-Time Benchmark v2.0
 * 
 * Shows progress in real-time with live statistics
 * Compares classical vs enhanced search performance
 * Optimized for Meteor Lake architecture
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <unistd.h>
#include "include/not_stisla.h"

/* Benchmark configuration - can be overridden via command line */
#define WARMUP_ITERATIONS 500
#define BENCHMARK_ITERATIONS 50000
#define LIVE_UPDATE_INTERVAL 1000

/* ANSI color codes */
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

/* Timing utilities */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Generate sorted test array */
static void generate_sorted_array(int64_t* arr, size_t n, int workload_type) {
    srand(42); /* Fixed seed for reproducibility */
    switch (workload_type) {
        case NOT_STISLA_WORKLOAD_TELEMETRY:
            for (size_t i = 0; i < n; i++) {
                arr[i] = (int64_t)i * 1000 + (int64_t)(rand() % 500);
            }
            break;
        case NOT_STISLA_WORKLOAD_IDS:
            for (size_t i = 0; i < n; i++) {
                arr[i] = (int64_t)i * 2;
            }
            break;
        default:
            for (size_t i = 0; i < n; i++) {
                arr[i] = (int64_t)i;
            }
            break;
    }
}

/* Live statistics display */
static void show_live_stats(int current, int total, const char* label, 
                           double running_avg, uint64_t min_ns, uint64_t max_ns) {
    int width = 30;
    int pos = (current * width) / total;
    
    printf("\r%s [", label);
    for (int i = 0; i < width; i++) {
        if (i < pos) printf("█");
        else if (i == pos) printf("▓");
        else printf("░");
    }
    printf("] %3d%% | " GREEN "%.1f ns" RESET " (min:%lu max:%lu)", 
           (current * 100) / total, running_avg, 
           (unsigned long)min_ns, (unsigned long)max_ns);
    fflush(stdout);
}

/* Run benchmark with real-time display */
static void run_benchmark(size_t array_size, int workload_type) {
    const char* workload_names[] = {"TELEMETRY", "IDS", "OFFSETS", "EVENTS"};
    
    printf("\n" BOLD CYAN "═══════════════════════════════════════════════════════════════\n" RESET);
    printf(BOLD "  Array Size: %zu  |  Workload: %s\n" RESET, 
           array_size, workload_type < 4 ? workload_names[workload_type] : "DEFAULT");
    printf(CYAN "═══════════════════════════════════════════════════════════════\n" RESET);

    /* Allocate and generate test data */
    int64_t* arr = malloc(array_size * sizeof(int64_t));
    if (!arr) {
        printf(RED "  ERROR: Failed to allocate array\n" RESET);
        return;
    }
    generate_sorted_array(arr, array_size, workload_type);

    /* Generate search keys (values that exist in the array) */
    size_t num_keys = 100;
    int64_t* keys = malloc(num_keys * sizeof(int64_t));
    for (size_t i = 0; i < num_keys; i++) {
        size_t idx = (i * 7) % array_size;
        keys[i] = arr[idx];
    }

    /* Create anchor tables */
    not_stisla_anchor_table_t* table_classical = not_stisla_anchor_table_create();
    not_stisla_anchor_table_t* table_enhanced = not_stisla_anchor_table_create();
    
    /* Initialize config */
    not_stisla_config_t config;
    not_stisla_config_init(&config, workload_type);
    config.quantum.enable_performance_tracking = 1;

    /* Warmup */
    printf("\n  " YELLOW "Warming up..." RESET);
    fflush(stdout);
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        int64_t key = keys[i % num_keys];
        not_stisla_search(arr, array_size, key, table_classical, 8);
        not_stisla_search_enhanced(arr, array_size, key, table_enhanced, &config);
    }
    printf(" " GREEN "Done" RESET "\n\n");

    /* Benchmark Classical */
    printf("  " BOLD "Classical NOT_STISLA:\n" RESET);
    uint64_t classical_total = 0;
    uint64_t classical_min = UINT64_MAX;
    uint64_t classical_max = 0;
    size_t classical_found = 0;

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        if ((i + 1) % LIVE_UPDATE_INTERVAL == 0 || i == BENCHMARK_ITERATIONS - 1) {
            double running_avg = (double)classical_total / (i + 1);
            show_live_stats(i + 1, BENCHMARK_ITERATIONS, "    Classical", running_avg, classical_min, classical_max);
        }
        
        int64_t key = keys[i % num_keys];
        uint64_t start = get_time_ns();
        not_stisla_result_t result = not_stisla_search(arr, array_size, key, table_classical, 8);
        uint64_t elapsed = get_time_ns() - start;
        
        classical_total += elapsed;
        if (elapsed < classical_min) classical_min = elapsed;
        if (elapsed > classical_max) classical_max = elapsed;
        if (result != NOT_STISLA_NOT_FOUND) classical_found++;
    }
    printf("\n");

    double classical_avg = (double)classical_total / BENCHMARK_ITERATIONS;
    printf("    " GREEN "Avg: %.2f ns" RESET "  |  Min: %lu ns  |  Max: %lu ns  |  Found: %zu/%d\n",
           classical_avg, (unsigned long)classical_min, (unsigned long)classical_max,
           classical_found, BENCHMARK_ITERATIONS);

    /* Benchmark Enhanced */
    printf("\n  " BOLD "Enhanced NOT_STISLA:\n" RESET);
    not_stisla_reset_performance_stats();
    
    uint64_t enhanced_total = 0;
    uint64_t enhanced_min = UINT64_MAX;
    uint64_t enhanced_max = 0;
    size_t enhanced_found = 0;

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        if ((i + 1) % LIVE_UPDATE_INTERVAL == 0 || i == BENCHMARK_ITERATIONS - 1) {
            double running_avg = (double)enhanced_total / (i + 1);
            show_live_stats(i + 1, BENCHMARK_ITERATIONS, "    Enhanced", running_avg, enhanced_min, enhanced_max);
        }
        
        int64_t key = keys[i % num_keys];
        uint64_t start = get_time_ns();
        not_stisla_result_t result = not_stisla_search_enhanced(arr, array_size, key, table_enhanced, &config);
        uint64_t elapsed = get_time_ns() - start;
        
        enhanced_total += elapsed;
        if (elapsed < enhanced_min) enhanced_min = elapsed;
        if (elapsed > enhanced_max) enhanced_max = elapsed;
        if (result != NOT_STISLA_NOT_FOUND) enhanced_found++;
    }
    printf("\n");

    double enhanced_avg = (double)enhanced_total / BENCHMARK_ITERATIONS;
    printf("    " GREEN "Avg: %.2f ns" RESET "  |  Min: %lu ns  |  Max: %lu ns  |  Found: %zu/%d\n",
           enhanced_avg, (unsigned long)enhanced_min, (unsigned long)enhanced_max,
           enhanced_found, BENCHMARK_ITERATIONS);

    /* Comparison */
    printf("\n  " BOLD "Comparison:\n" RESET);
    double speedup = classical_avg / enhanced_avg;
    double overhead_pct = ((enhanced_avg - classical_avg) / classical_avg) * 100.0;
    
    if (speedup >= 1.0) {
        printf("    Speedup: " GREEN "%.2fx" RESET " (Enhanced is %.1f%% faster)\n", 
               speedup, (speedup - 1.0) * 100.0);
    } else {
        printf("    Overhead: " YELLOW "%.1f%%" RESET " (Enhanced is %.2fx slower)\n", 
               overhead_pct, 1.0 / speedup);
    }

    /* Batch search (sequential) configuration */
    const size_t batch_size = 256;
    size_t batch_runs = BENCHMARK_ITERATIONS / 100;
    if (batch_runs == 0) batch_runs = 1;
    not_stisla_batch_item_t* batch_items = calloc(batch_size, sizeof(not_stisla_batch_item_t));
    not_stisla_anchor_table_t* batch_table = not_stisla_anchor_table_create();
    not_stisla_anchor_table_t* parallel_table = not_stisla_anchor_table_create();
    not_stisla_parallel_config_t parallel_config = {0, 0, 16};

    double batch_total_per_key_ns = 0.0;
    double batch_min_per_key_ns = DBL_MAX;
    double batch_max_per_key_ns = 0.0;
    size_t batch_found_total = 0;

    if (!batch_items || !batch_table || !parallel_table) {
        printf(RED "  ERROR: Unable to allocate anchors for batch/parallel tests\n" RESET);
    } else {
        printf("\n  " BOLD "Batch NOT_STISLA (Sequential sweep):\n" RESET);
        size_t batch_display_interval = batch_runs / 8;
        if (batch_display_interval == 0) batch_display_interval = 1;

        for (size_t run = 0; run < batch_runs; ++run) {
            for (size_t j = 0; j < batch_size; ++j) {
                size_t idx = (run * batch_size + j) % num_keys;
                batch_items[j].key = keys[idx];
                batch_items[j].result = NOT_STISLA_NOT_FOUND;
                batch_items[j].ordinal = j;
            }

            uint64_t start = get_time_ns();
            size_t found = not_stisla_search_batch(arr, array_size, batch_items, batch_size, batch_table, 8);
            uint64_t elapsed = get_time_ns() - start;

            double per_key = (double)elapsed / batch_size;
            batch_total_per_key_ns += per_key;
            batch_min_per_key_ns = fmin(batch_min_per_key_ns, per_key);
            batch_max_per_key_ns = fmax(batch_max_per_key_ns, per_key);
            batch_found_total += found;

            if ((run + 1) % batch_display_interval == 0 || run == batch_runs - 1) {
                double running_avg = batch_total_per_key_ns / (run + 1);
                show_live_stats(run + 1, batch_runs, "    Batch Seq", running_avg,
                                (uint64_t)batch_min_per_key_ns, (uint64_t)batch_max_per_key_ns);
            }
        }

        printf("\n    " GREEN "Avg per key: %.2f ns" RESET
               "  |  Min: %.2f ns  |  Max: %.2f ns  |  Found: %zu/%zu\n",
               batch_total_per_key_ns / batch_runs,
               batch_min_per_key_ns,
               batch_max_per_key_ns,
               batch_found_total,
               batch_runs * batch_size);

        double parallel_total_per_key_ns = 0.0;
        double parallel_min_per_key_ns = DBL_MAX;
        double parallel_max_per_key_ns = 0.0;
        size_t parallel_found_total = 0;

        printf("\n  " BOLD "Batch NOT_STISLA (Parallel OpenMP):\n" RESET);
        for (size_t run = 0; run < batch_runs; ++run) {
            for (size_t j = 0; j < batch_size; ++j) {
                size_t idx = (run * batch_size + j) % num_keys;
                batch_items[j].key = keys[idx];
                batch_items[j].result = NOT_STISLA_NOT_FOUND;
                batch_items[j].ordinal = j;
            }

            uint64_t start = get_time_ns();
            size_t found = not_stisla_search_parallel(arr, array_size, batch_items, batch_size,
                                                      parallel_table, 8, &parallel_config);
            uint64_t elapsed = get_time_ns() - start;

            double per_key = (double)elapsed / batch_size;
            parallel_total_per_key_ns += per_key;
            parallel_min_per_key_ns = fmin(parallel_min_per_key_ns, per_key);
            parallel_max_per_key_ns = fmax(parallel_max_per_key_ns, per_key);
            parallel_found_total += found;

            if ((run + 1) % batch_display_interval == 0 || run == batch_runs - 1) {
                double running_avg = parallel_total_per_key_ns / (run + 1);
                show_live_stats(run + 1, batch_runs, "    Batch Parallel", running_avg,
                                (uint64_t)parallel_min_per_key_ns, (uint64_t)parallel_max_per_key_ns);
            }
        }

        printf("\n    " GREEN "Avg per key: %.2f ns" RESET
               "  |  Min: %.2f ns  |  Max: %.2f ns  |  Found: %zu/%zu\n",
               parallel_total_per_key_ns / batch_runs,
               parallel_min_per_key_ns,
               parallel_max_per_key_ns,
               parallel_found_total,
               batch_runs * batch_size);
    }

    /* Cleanup */
    free(batch_items);
    not_stisla_anchor_table_destroy(batch_table);
    not_stisla_anchor_table_destroy(parallel_table);
    not_stisla_anchor_table_destroy(table_classical);
    not_stisla_anchor_table_destroy(table_enhanced);
    free(keys);
    free(arr);
}

int main(int argc, char** argv) {
    printf("\n" BOLD CYAN "╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║          NOT_STISLA Real-Time Benchmark                       ║\n");
    printf("║          QIHSE-Enhanced Search Algorithm                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n" RESET);
    
    printf("\n  Iterations per test: %d\n", BENCHMARK_ITERATIONS);
    printf("  Warmup iterations:   %d\n", WARMUP_ITERATIONS);
    printf("  Architecture:        meteor lake (AVX2/AVX512)\n");

    int workload_type = NOT_STISLA_WORKLOAD_TELEMETRY;
    if (argc > 1) {
        workload_type = atoi(argv[1]);
        if (workload_type < 0 || workload_type > 3) {
            workload_type = NOT_STISLA_WORKLOAD_TELEMETRY;
        }
    }

    /* Test multiple array sizes */
    size_t sizes[] = {100, 1000, 10000, 100000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        run_benchmark(sizes[i], workload_type);
    }

    printf("\n" BOLD CYAN "═══════════════════════════════════════════════════════════════\n" RESET);
    printf(BOLD GREEN "  Benchmark Complete!\n" RESET);
    printf(CYAN "═══════════════════════════════════════════════════════════════\n\n" RESET);

    return 0;
}
