/* Harmless per-rank report fixture; no MPI library. Usage:
 *   rank_probe CALL_RANKS [normal|noreport|sleep]
 * CALL_RANKS is a comma list of ranks (or "all"/"none") that make one ordinary
 * 32x32x32 zgemm_ call; other ranks never call ZGEMM. noreport leaves through
 * _exit (as a killed rank would); sleep waits to be signalled. */
#include "vmb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int selected(const char *list, int rank)
{
    if (strcmp(list, "all") == 0)
        return 1;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ","))
        if (strcmp(t, "none") != 0 && atoi(t) == rank)
            return 1;
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
        return 2;
    const char *r = getenv("OMPI_COMM_WORLD_RANK");
    int rank = r ? atoi(r) : 0;
    const char *how = argc == 3 ? argv[2] : "normal";
    if (selected(argv[1], rank)) {
        enum { N = 32 };
        static double a[2 * N * N], b[2 * N * N], c[2 * N * N];
        for (int i = 0; i < 2 * N * N; i++) {
            a[i] = (double)((i * 7 + rank) % 13) / 13.0 - 0.5;
            b[i] = (double)((i * 5 + 3) % 11) / 11.0 - 0.5;
        }
        const double alpha[2] = {1, 0}, beta[2] = {0, 0};
        int n = N;
        zgemm_("N", "N", &n, &n, &n, alpha, a, &n, b, &n, beta, c, &n, 1, 1);
    }
    if (strcmp(how, "noreport") == 0)
        _exit(0);
    if (strcmp(how, "sleep") == 0)
        pause();
    printf("rank %d done\n", rank);
    return 0;
}
