#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

static void swap(double *x, double *y);
static void swapint(int *x, int *y);
static int isdigits(char *s);
static void help(void);
static void memerr(void);

static int CreateSystem(int argc, char *argv[], double ***pa, double **pf, int *n);
static void PrintSystem(double **a, double *f, int n);
static int ReadFile(FILE *in, double ***pa, double **pf, int n);

static void FreeMatrix(double **a, int n);
static void FillMatrix(double **a, int n);
static void FillMatrix1(double **a, int n);
static void PrintMatrix(double **a, int n);
static double **AllocMatrix(int n);
static double **CopyMatrix(double **a, int n);

static void FillVector(double *f, int n);
static void PrintVector(double *f, int n);
static double *CopyVector(double *f, int n);

static double **MatMatMult(double **a, double **b, int n);
static double *MatVecMult(double **a, double *f, int n);
static void Transpon(double **a, int n);

static double *Gauss(double **a, double *f, int n);
static double *GaussChoice(double **a, double *f, int n);
static void Seidel(double **a, double *f, int n);
static void UpRelax(double **a, double *f, double w, int n);

static double det(double **a, int n);
static double MatNorm(double **a, int n);
static double VecNorm(double *f, int n);
static double **InvMat(double **a, int n);

static void swap(double *x, double *y)
{
    double r = *x;
    *x = *y;
    *y = r;
}

static void swapint(int *x, int *y)
{
    int r = *x;
    *x = *y;
    *y = r;
}

//check whether the string contains nothing but the digits' symbols
static int isdigits(char *s)
{
    while (*s) {
        if (*s < '0' || *s > '9')
            return 0;
        s++;
    }

    return 1;
}

static void help(void)
{
    printf("./gauss <PARAM1> [<PARAM2>] [<PARAM3>]\n");
    printf("PARAM1: -help (or -h), -input (-i), or -formula (-f)\n");
    printf("PARAM2:\n1) matrix size n (integer number) if PARAM1=-formula (-f)\n");
    printf("2) input file path if PARAM1=-input (-i)\n");
    printf("3) none if PARAM1=-help (-h)\n");
    printf("PARAM3:\n1) x parameter (for calculating the system) if PARAM1=-formula (-f)\n");
    printf("2) none otherwise\n");
    printf("\nPress ENTER to finish\n");
}

static void memerr(void)
{
    printf("Error: out of memory\n");
    _exit(1);
}

//handling command line arguments
//reading or calculating the matrix and the vector
//return value: 0 - the system of linear equations successfully created, 1 - otherwise
static int CreateSystem(int argc, char *argv[], double ***pa, double **pf, int *pn)
{
    int n;
    double **a, *f;

    if (argc < 2) {
        printf("Error: first parameter missing\n");
        return 1;
    }

    if (strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0) {
        help();
        getchar();
        exit(0);

    } else if (strcmp(argv[1], "-input") == 0 || strcmp(argv[1], "-i") == 0) {
        if (argc < 3) {
            printf("Error: input file path missing\n");
            return 1;
        }

        FILE *in = fopen(argv[2], "r");

        if (in == NULL) {
            printf("Error: input file not found\n");
            return 1;
        }

        if (fscanf(in, "%d", &n) != 1) {
            printf("Error: data missing\n");
            fclose(in);
            return 1;
        }

        int err = ReadFile(in, &a, &f, n);

        fclose(in);
        if (err) {
            if (err == 2)
                memerr();
            else
                printf("Error: data missing\n");

            return 1;
        }

    } else if (strcmp(argv[1], "-formula") == 0 || strcmp(argv[1], "-f") == 0) {

        if (argc < 3 || !isdigits(argv[2])) {
            printf("Error: matrix size missing\n");
            return 1;
        }
        sscanf(argv[2], "%d", &n);

        f = calloc(n, sizeof(*f));
        a = AllocMatrix(n);

        if (!f || !a) {

            if (f) free(f);
            if (a) FreeMatrix(a, n);

            printf("Error: not enough memory\n");
            return 1;
        }

        FillMatrix(a, n);
        FillVector(f, n);

    } else {
        printf("Error: wrong first parameter\n");
        return 1;
    }

    *pa = a;
    *pf = f;
    *pn = n;
    return 0;
}

static double **AllocMatrix(int n)
{
    double **a = calloc(n, sizeof(*a));

    if (a == NULL) {
        return NULL;
    }

    for (int i = 0; i < n; ++i) {
        a[i] = calloc(n, sizeof(**a));
        if (a[i] == NULL) {
            for (int j = 0; j < i; ++j)
                free(a[j]);
            free(a);
            return NULL;
        }
    }

    return a;
}

static void PrintSystem(double **a, double *f, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%12.6f", a[i][j]);
        printf(" | %12.6f\n", f[i]);
    }
}

static double **CopyMatrix(double **a, int n)
{
    double **res = calloc(n, sizeof(*res));
    for (int i = 0; i < n; ++i) {
        res[i] = calloc(n, sizeof(**res));
        for (int j = 0; j < n; ++j) res[i][j] = a[i][j];
    }
    return res;
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

//spoils the matrix
static void Transpon(double **a, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            swap(&a[i][j], &a[j][i]);
}

static void FreeMatrix(double **a, int n)
{
    for (int i = 0; i < n; ++i)
        free(a[i]);
    free(a);
}

static void PrintMatrix(double **a, int n)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            printf("%12.6f", a[i][j]);
        printf("\n");
    }
}

//fill the matrix according to the formula
static void FillMatrix(double **a, int n)
{
    double m;
    printf("Type m parameter for filling the matrix:\n");
    scanf("%lf", &m);
    m = 1.001 - 2 * m / 1000;

    double pow1 = m * m, pow2 = (m - 1) * (m - 1);
    int maxpow = 2 * n, i;

    for (int pow = 2; pow <= maxpow; ++pow) {
        for (int j = 1; j < pow; ++j) {
            i = pow - j;
            if (i <= n && j <= n)
                a[i - 1][j - 1] = i == j ? pow2 : pow1 + 0.1 * (j - i);
        }
        pow1 *= m;
        pow2 *= m - 1;
    }
}

static void FillMatrix1(double **a, int n)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            a[i][j] = rand() % (i * j + 2) * 1.0 / (rand() % (j + 1) + 1) ;
}

//fill the vector according to the formula
static void FillVector(double *f, int n)
{
    double x;
    printf("Type x parameter for filling the vector:\n");
    scanf("%lf", &x);
    for (int i = 1; i <= n; ++i)
        f[i - 1] = x * exp(x / i) * cos(x / i);
}

static void PrintVector(double *f, int n)
{
    for (int i = 0; i < n; ++i)
        printf(" x%d = %8.6f", i + 1, f[i]);
}

static double *CopyVector(double *f, int n)
{
    double *cpy = calloc(n, sizeof(*cpy));
    if (cpy == NULL) return NULL;

    for (int i = 0; i < n; ++i) cpy[i] = f[i];
    return cpy;
}

//allocate memory for matrix and vector,
//read them from the file and write the pointer
//to them to pa and pf
//return value: 0 - data succsefully read and written to p and pf,
//1 - data missing in the input file, 2 - not enough memory for the system
static int ReadFile(FILE *in, double ***pa, double **pf, int n)
{
    double *f = calloc(n, sizeof(*f));
    double **a = AllocMatrix(n);

    if (!f || !a) {

        if (f) free(f);
        if (a) FreeMatrix(a, n);

        return 2;
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (fscanf(in, "%lf", &a[i][j]) != 1) {
                FreeMatrix(a, n);
                free(f);
                return 1;
            }

    for (int i = 0; i < n; ++i)
        if (fscanf(in, "%lf", &f[i]) != 1) {
            FreeMatrix(a, n);
            free(f);
            return 1;
        }

    *pa = a;
    *pf = f;

    return 0;
}

//converting the system to the triangular form
//computing the solution and returning it
static double *Gauss(double **a, double *f, int n)
{
    double div;
    for (int i = 0; i < n; ++i) {
        if (a[i][i] == 0)
            for (int j = i + 1; j < n; ++j)
                if (a[j][i] != 0) {
                    for (int c = i; c < n; ++c)
                        swap(&a[j][c], &a[i][c]);
                    swap(&f[j], &f[i]);
                }

        div = a[i][i];
        for (int j = i; j < n; ++j) {
            a[i][j] /= div;
        }
        f[i] /= div;

        for (int j = i + 1; j < n; ++j) {
            for (int c = i + 1; c < n; ++c)
                a[j][c] -= a[j][i] * a[i][c];
            f[j] -= a[j][i] * f[i];
            a[j][i] = 0;

        }
    }

    double *res = calloc(n, sizeof(*res));
    if (res == NULL) return NULL;
    res[n - 1] = f[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        res[i] = f[i];
        for (int j = n - 1; j > i; --j) {
            res[i] -= res[j] * a[i][j];
        }
    }

    return res;
}

static double *GaussChoice(double **a, double *f, int n)
{
    double div, max;
    int colmax;
    int *pos = calloc(n, sizeof(*pos));
    for (int i = 0; i < n; ++i) pos[i] = i;

    for (int i = 0; i < n; ++i) {
        max = fabs(a[i][i]);
        colmax = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(a[i][j]) > max) {
                max = fabs(a[i][j]);
                colmax = j;
            }

        if (i != colmax) {
            for (int c = 0; c < n; ++c)
                swap(&a[c][colmax], &a[c][i]);
            swapint(&pos[i], &pos[colmax]);
        }

        div = a[i][i];
        for (int j = i; j < n; ++j) {
            a[i][j] /= div;
        }
        f[i] /= div;

        for (int j = i + 1; j < n; ++j) {
            for (int c = i + 1; c < n; ++c)
                a[j][c] -= a[j][i] * a[i][c];
            f[j] -= a[j][i] * f[i];
            a[j][i] = 0;

        }
    }

    double *res = calloc(n, sizeof(*res));
    if (res == NULL) return NULL;
    for (int i = n - 2; i >= 0; --i) {
        for (int j = n - 1; j > i; --j) {
            f[i] -= f[j] * a[i][j];
        }
    }

    for (int i = 0; i < n; ++i) res[pos[i]] = f[i];

    free(pos);
    return res;
}

static void Seidel(double **a, double *f, int n)
{
    //the matrix A^T * A is always self-adjoint and positive definite
    //Ax=f => A^T*Ax = A^T*f
    double eps;
    printf("Type eps: ");
    scanf("%lf", &eps);
    eps /= 1000;

    double **trans = CopyMatrix(a, n);
    double *res = f;
    Transpon(trans, n);
    a = MatMatMult(trans, a, n);
    f = MatVecMult(trans, f, n);

    double *prev = calloc(n, sizeof(*prev)),
           *next = calloc(n, sizeof(*next)),
           *diff = calloc(n, sizeof(*next)), s;

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
    FreeMatrix(trans, n);
}

static void UpRelax(double **a, double *f, double w, int n)
{
    //the matrix A^T * A is always self-adjoint and positive definite
    //Ax=f => A^T*Ax = A^T*f
    double eps;
    printf("Type eps: ");
    scanf("%lf", &eps);
    eps /= 1000;
    int iter = 1;

    double **trans = CopyMatrix(a, n);
    double *res = f;
    Transpon(trans, n);
    a = MatMatMult(trans, a, n);
    f = MatVecMult(trans, f, n);

    double *prev = calloc(n, sizeof(*prev)),
           *next = calloc(n, sizeof(*next)),
           *diff = calloc(n, sizeof(*next)), s;

    do {
        for (int i = 0; i < n; ++i) {
            s = 0;
            for (int j = 0; j < i; ++j) {
                s += next[j] * a[i][j];
            }
            for (int j = i; j < n; ++j) {
                s += prev[j] * a[i][j];
            }
            next[i] = prev[i] + w * (f[i] - s) / a[i][i];
        }
        for (int i = 0; i < n; ++i) {
            diff[i] = prev[i] - next[i];
            prev[i] = next[i];
        }
        iter++;
    } while (VecNorm(diff, n) > eps);

    printf("\n\n%d iterations, w = %8.3f\n\n", iter, w);

    for (int i = 0; i < n; ++i) res[i] = next[i];
    free(prev);
    free(next);
    free(f);
    FreeMatrix(a, n);
    FreeMatrix(trans, n);
}

static double ResidualNorm(double **a, double *f, double *x, int n)
{
    double *tmp = MatVecMult(a, x, n);
    for (int i = 0; i < n; ++i) tmp[i] -= f[i];
    double norm = VecNorm(tmp, n);
    free(tmp);
    return norm;
}

static double **InvMat(double **a, int n)
{
    double **inv = calloc(n, sizeof(*inv));
    if (inv == NULL) memerr();
    for (int i = 0; i < n; ++i) {
        inv[i] = calloc(n, sizeof(**inv));
        if (inv[i] == NULL) memerr();
        inv[i][i] = 1;
    }

    double div, max;
    int strmax;
    for (int i = 0; i < n; ++i) {
        max = fabs(a[i][i]);
        strmax = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(a[j][i]) > max) {
                max = fabs(a[j][i]);
                strmax = j;
            }

        for (int c = 0; c < n; ++c) {
            swap(&a[strmax][c], &a[i][c]);
            swap(&inv[strmax][c], &inv[i][c]);
        }

        div = a[i][i];
        for (int j = 0; j < n; ++j) {
            a[i][j] /= div;
            inv[i][j] /= div;
        }

        //printf("\n");
        //PrintMatrix(a,n);

        for (int j = i + 1; j < n; ++j) {
            for (int c = i + 1; c < n; ++c) {
                a[j][c] -= a[j][i] * a[i][c];
            }
            for (int c = 0; c < n; ++c) {
                inv[j][c] -= a[j][i] * inv[i][c];
            }
            a[j][i] = 0;
        }

    }

    for (int col = 0; col < n; ++col)
        for (int i = 0; i < col; ++i) {
            div = a[i][col];
            for (int j = 0; j < n; ++j) {
                a[i][j] -= a[col][j] * div;
                inv[i][j] -= inv[col][j] * div;
            }
        }

    return inv;
}

static double MatNorm(double **a, int n)
{
    double norm = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            norm += a[i][j] * a[i][j];
    return sqrt(norm);
}

static double VecNorm(double *f, int n)
{
    double norm = 0;
    for (int i = 0; i < n; ++i) norm += f[i] * f[i];
    return sqrt(norm);
}

//matrix determinant
static double det(double **a, int n)
{
    int sign = 1, flag = 1, i, j, c;
    double first;
    for (i = 0; i < n - 1; ++i) {
        if (a[i][i] == 0) {
            for (j = 0; j < n; ++j)
                if (a[j][i] != 0) {
                    flag = 0;
                    break;
                }
            if (flag++) return 0; //no non-zero elements in the column
            for (c = 0; c < n; ++c)
                swap(&a[i][c], &a[j][c]);
            sign = -sign;
        }
        for (j = i + 1; j < n; ++j) {
            first = a[j][i];
            for (c = i; c < n; ++c) {
                a[j][c] -= first * a[i][c] / a[i][i];
            }
        }
    }

    double res = 1;
    for (int i = 0; i < n; ++i) res *= a[i][i];

    return res * sign;
}

int main(int argc, char *argv[])
{
    setbuf(stdout, NULL);
    int n;
    double **a, *f;

    if (CreateSystem(argc, argv, &a, &f, &n)) {
        return 1;
    }

    int fd = open("out.txt", O_WRONLY | O_APPEND);
    dup2(fd, STDOUT_FILENO);


    printf("The system of linear equations:\n\n");
    PrintSystem(a, f, n);

    pid_t pid = fork();

    if (pid < 0)
        memerr();
    if (!pid) {
        double **acp = CopyMatrix(a, n);
        double *fcp = CopyVector(f, n);
        double *res = Gauss(a, f, n);
        if (res == NULL || acp == NULL || fcp == NULL) memerr();
        printf("\nGauss method:\nThe solution vector is:\n");
        PrintVector(res, n);
        printf("\nResidual norm: %30.20f\n", ResidualNorm(acp, fcp, res, n));
        FreeMatrix(a, n);
        free(f);
        FreeMatrix(acp, n);
        free(fcp);
        free(res);
        return 0;
    }
    wait(NULL);

    if ((pid = fork()) < 0)
        memerr();
    if (!pid) {
        double **acp = CopyMatrix(a, n);
        double *fcp = CopyVector(f, n);
        double *res = GaussChoice(a, f, n);
        if (res == NULL || acp == NULL || fcp == NULL) memerr();
        printf("\n\nGauss method (choosing maximum value in the string):\nThe solution vector is:\n");
        PrintVector(res, n);
        printf("\nResidual norm: %30.20f\n", ResidualNorm(acp, fcp, res, n));
        FreeMatrix(a, n);
        free(f);
        FreeMatrix(acp, n);
        free(fcp);
        free(res);
        return 0;
    }
    wait(NULL);

    if ((pid = fork()) < 0)
        memerr();
    if (!pid) {
        free(f);

        printf("\n\ndet A = %20.15f\n", det(a, n));

        FreeMatrix(a, n);
        return 0;
    }
    wait(NULL);

    if ((pid = fork()) < 0)
        memerr();
    if (!pid) {
        free(f);

        double matnorm = MatNorm(a, n);
        double **inv = InvMat(a, n);
        if (inv == NULL)
            memerr();

        double invnorm = MatNorm(inv, n);

        printf("\nThe inverse matrix is:\n");
        PrintMatrix(inv, n);
        printf("\nCondition number = %8.3f\n\n", matnorm * invnorm);

        FreeMatrix(a, n);
        FreeMatrix(inv, n);
        return 0;
    }
    wait(NULL);

    if ((pid = fork()) < 0)
        memerr();
    if (!pid) {
        double **acp = CopyMatrix(a, n);
        double *fcp = CopyVector(f, n);
        if (acp == NULL || fcp == NULL) memerr();
        printf("Seidel method:\n");
        Seidel(a, f, n);
        printf("The solution vector is:\n");
        PrintVector(f, n);
        printf("\nResidual norm: %8.6f\n", ResidualNorm(acp, fcp, f, n));
        FreeMatrix(a, n);
        free(f);
        FreeMatrix(acp, n);
        free(fcp);
        return 0;
    }
    wait(NULL);

    if ((pid = fork()) < 0)
        memerr();
    if (!pid) {
        double **acp = CopyMatrix(a, n);
        double *fcp = CopyVector(f, n);
        if (acp == NULL || fcp == NULL) memerr();
        double w;
        printf("\nUpper relaxation method:\n");
        printf("Type w parameter:\n");
        scanf("%lf", &w);
        UpRelax(a, f, w, n);
        printf("The solution vector is:\n");
        PrintVector(f, n);
        printf("\nResidual norm: %8.6f\n", ResidualNorm(acp, fcp, f, n));
        FreeMatrix(a, n);
        free(f);
        FreeMatrix(acp, n);
        free(fcp);
        return 0;
    }
    wait(NULL);

    FreeMatrix(a, n);
    free(f);
    return 0;
}
