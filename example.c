#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>

static void memerr(void)
{
    perror("Error: out of memory");
    _exit(1);
}

static double **MatMatMult(double **a, double **b, int n)
{
    double **res = calloc(n, sizeof(*res));
    if (res == NULL) return NULL;
    for (int i = 0; i < n; ++i) {
        res[i] = calloc(n, sizeof(**res));
        if (res[i] == NULL) return NULL;
        for (int j = 0; j < n; ++j) {
            for (int c = 0; c < n; ++c)
                res[i][j] += a[i][c] * b[c][j];
        }
    }

    return res;
}

static double *MatVecMult(double **a, double *f, int n)
{
    double *res = calloc(n, sizeof(*res));
    if (res == NULL) return NULL;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            res[i] += a[i][j] * f[j];

    return res;
}

static void Transpon(double **a, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            swap(&a[i][j], &a[j][i]);
}

//solving the system of linear equations Ax=f (result in f)
static void Seidel(double **a, double *f, int n)
{
    //the matrix A^T * A is always self-adjoint and positive definite
    //Ax=f => A^T*Ax = A^T*f
    double eps;
    printf("Type eps: ");
    scanf("%lf", &eps);
    eps /= 1000;

    double **trans = CopyMatrix(a, n);
    double **cpy = CopyMatrix(a, n);
    if (trans == NULL || cpy == NULL) memerr();
    double *res = f;
    Transpon(trans, n);
    a = MatMatMult(trans, cpy, n);
    f = MatVecMult(trans, f, n);
    if (a == NULL || f == NULL) memerr();

    double *prev = calloc(n, sizeof(*prev)),
           *next = calloc(n, sizeof(*next)),
           *diff = calloc(n, sizeof(*next)), s;

    if (prev == NULL || next == NULL || diff == NULL) memerr();

    do {
        for (int i = 0; i < n; ++i) {
            s = 0;
            for (int j = 0; j < i; ++j) {
                s += next[j] * a[i][j];
            }
            for (int j = i + 1; j < n; ++j) {
                s += prev[j] * a[i][j];
            }
            next[i] = (f[i] - s) / a[i][i];
        }
        for (int i = 0; i < n; ++i) {
            diff[i] = prev[i] - next[i];
            prev[i] = next[i];
        }

    } while (VecNorm(diff, n) > eps);

    for (int i = 0; i < n; ++i) res[i] = next[i];
    free(prev);
    free(next);
    free(f);
    FreeMatrix(a, n);
    FreeMatrix(cpy, n);
    FreeMatrix(trans, n);
}
