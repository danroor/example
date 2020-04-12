#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

//4: y-yx,  x0 = 0, y0 = 5
//S: 5e^(x * (1 - x / 2) )
//14:
//f1: e^-(u^2 + v^2) + 2x
//f2: 2u^2 + v
//x0 = 0,  u0 = 0.5, v0 = 1

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

static void memerr(void)
{
    printf("Error: out of memory\n");
    _exit(1);
}

static double func1(double x, double y)
{
    return y - x * y;
}

// 3y/2x + 9x^4/2(y+3)^2
static double func2(double x, double y)
{
    return (3.0 / 2) * (y / x) - 3 * x * x / (2 * (y - 1) * (y - 1));
}

static double func3(double x, double y)
{
    return y / x + x * sqrt(1 - y * y / (x * x));
}

static double func4(double x, double y)
{
    return exp(x) - 3 * y;
}

//e^-(u^2 + v^2) + 2x
static double sysf11(double x, double u, double v)
{
    return exp(-u * u - v * v) + 2 * x;
}

//2u^2 + v
static double sysf12(double x, double u, double v)
{
    return 2 * u * u + v;
}

static double sysf21(double x, double u, double v)
{
    return (u - v) / x;
}

static double sysf22(double x, double u, double v)
{
    return 2 * v / x + u * x;
}

static double sysf31(double x, double u, double v)
{
    return 1 / (2 * u);
}

static double sysf32(double x, double u, double v)
{
    return v;
}

//(u - 3x)ln2  + 3
static double sysf41(double x, double u, double v)
{
    return (u - 3 * x) * log(2) + 3;
}

static double sysf42(double x, double u, double v)
{
    return 1 / x;
}

//2e^(v-3)
static double sysf51(double x, double u, double v)
{
    return 2 * exp(v - 3);
}

//u - 2e^x + 1
static double sysf52(double x, double u, double v)
{
    return u - 2 * exp(x) + 1;
}

static void PrintFunc(double *f, double x, double h, int n)
{
    for (int i = 0; i < n; ++i) {
        printf("%15.6f %15.6f\n", x, f[i]);
        x += h;
    }
}

//Runge-Kutta second-order method for equation
//returns an array of n values of the approximating function
//x0 = a (left border)
static double *RK2(double a, double b, double yy0, int n, double alpha, double (*f)(double, double))
{
    double *y = calloc(n + 1, sizeof(*y));
    if (y == NULL) return NULL;

    y[0] = yy0;
    double h = (b - a) / n,
           x = a, ff;

    for (int i = 0; i < n; ++i) {
        //y_i+1 = y_i + ((1 - a)f(x_i, y_i) + a*f( x_i + h/2a, y_i + h/2a * f(x_i, y_i) ) )h
        //a - alpha
        ff = f(x, y[i]);
        //(1 - alpha) * ff + alpha * f(x + h / (2 * alpha), y[i] + h * ff / (2 * alpha) ) );
        y[i + 1] = y[i] + ( (1 - alpha) * ff + alpha * f(x + h / (2 * alpha), y[i] + h * ff / (2 * alpha) ) ) * h;
        x += h;
    }
    return y;
}

//Runge-Kutta fourth-order method for equation
//returns an array of n values of the approximating function
//x0 = a (left border)
static double *RK4(double a, double b, double yy0, int n, double (*f)(double, double))
{
    double *y = calloc(n + 1, sizeof(*y));
    if (y == NULL) return NULL;

    y[0] = yy0;
    double h = (b - a) / n,
           x = a, k1, k2, k3, k4;

    for (int i = 0; i < n; ++i) {
        //y_i+1 = y_i + h/6(k1 + 2k2 + 2k3 + k4)
        //k1 = f(x_i, y_i)
        //k2 = f(x_i + h/2, y_i + k1*h/2)
        //k3 = f(x_i + h/2, y_i + k2*h/2)
        //k4 = f(x_i + h, y_i + hk3)

        k1 = f(x, y[i]);
        k2 = f(x + h / 2, y[i] + k1 * h / 2);
        k3 = f(x + h / 2, y[i] + k2 * h / 2);
        k4 = f(x + h, y[i] + k3 * h);

        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        x += h;
    }

    return y;
}

//Runge-Kutta second-order method for system
//returns two arrays of n values of the approximating functions
//x0 = a (left border)
static double **RKS2(double a, double b, double u0, double v0, int n, double alpha, double (*f1)(double, double, double), double (*f2)(double, double, double))
{
    double **uv = calloc(2, sizeof(*uv));
    if (uv == NULL) return NULL;

    //u
    uv[0] = calloc(n + 1, sizeof(*uv));
    if (uv[0] == NULL) {
        free(uv);
        return NULL;
    }
    //v
    uv[1] = calloc(n + 1, sizeof(*uv));
    if (uv[1] == NULL) {
        free(uv[0]);
        free(uv);
        return NULL;
    }

    uv[0][0] = u0;
    uv[1][0] = v0;
    double h = (b - a) / n,
           x = a, ff1, ff2;

    for (int i = 0; i < n; ++i) {
        //u_i+1 = y_i + ((1 - alpha)f1(x_i, u_i, v_i) + alpha*f1(x_i + h/2alpha, u_i + h/2alpha * f1(x_i, u_i, v_i), v_i + h/2alpha * f2(x_i, y_i, v_i) ) )h
        //v_i+1 = v_i + ((1 - alpha)f2(x_i, u_i, v_i) + alpha*f2(x_i + h/2alpha, u_i + h/2alpha * f1(x_i, u_i, v_i), v_i + h/2alpha * f2(x_i, y_i, v_i) ) )h
        ff1 = f1(x, uv[0][i], uv[1][i]);
        ff2 = f2(x, uv[0][i], uv[1][i]);
        uv[0][i + 1] = uv[0][i] + ( (1 - alpha) * ff1 + alpha * f1(x + h / (2 * alpha),
                    uv[0][i] + h * ff1 / (2 * alpha), uv[1][i] + h * ff2 / (2 * alpha) ) ) * h;
        uv[1][i + 1] = uv[1][i] + ( (1 - alpha) * ff2 + alpha * f2(x + h / (2 * alpha),
                    uv[0][i] + h * ff1 / (2 * alpha), uv[1][i] + h * ff2 / (2 * alpha) ) ) * h;
        x += h;
    }

    return uv;
}

//Runge-Kutta fourth-order method for system
//returns an array of n values of the approximating function
//x0 = a (left border)
static double **RKS4(double a, double b, double u0, double v0, int n, double (*f1)(double, double, double), double (*f2)(double, double, double))
{
    double **uv = calloc(2, sizeof(*uv));
    if (uv == NULL) return NULL;

    //u
    uv[0] = calloc(n + 1, sizeof(*uv));
    if (uv[0] == NULL) {
        free(uv);
        return NULL;
    }
    //v
    uv[1] = calloc(n + 1, sizeof(*uv));
    if (uv[1] == NULL) {
        free(uv[0]);
        free(uv);
        return NULL;
    }

    uv[0][0] = u0;
    uv[1][0] = v0;
    double h = (b - a) / n,
           x = a, k1, k2, k3, k4, l1, l2, l3, l4;

    for (int i = 0; i < n; ++i) {
        //u_i+1 = u_i + h/6(k1 + 2k2 + 2k3 + k4)
        //v_i+1 = v_i + h/6(l1 + 2l2 + 2l3 + l4)

        //k1 = f1(x_i, u_i, v_i)
        //l1 = f2(x_i, u_i, v_i)
        k1 = f1(x, uv[0][i], uv[1][i]);
        l1 = f2(x, uv[0][i], uv[1][i]);

        //k2 = f1(x_i + h/2, u_i + k1*h/2, v_i + l1*h/2)
        //l2 = f2(x_i + h/2, u_i + k1*h/2, v_i + l1*h/2)
        k2 = f1(x + h / 2, uv[0][i] + k1 * h / 2, uv[1][i] + l1 * h / 2);
        l2 = f2(x + h / 2, uv[0][i] + k1 * h / 2, uv[1][i] + l1 * h / 2);

        //k3 = f1(x_i + h/2, u_i + k2*h/2, v_i + l2*h/2)
        //l3 = f2(x_i + h/2, u_i + k2*h/2, v_i + l2*h/2)
        k3 = f1(x + h / 2, uv[0][i] + k2 * h / 2, uv[1][i] + l2 * h / 2);
        l3 = f2(x + h / 2, uv[0][i] + k2 * h / 2, uv[1][i] + l2 * h / 2);

        //k4 = f1(x_i + h, u_i + k3*h, v_i + l3*h)
        //l4 = f2(x_i + h, u_i + k3*h, v_i + l3*h)
        k4 = f1(x + h, uv[0][i] + k3 * h, uv[1][i] + l3 * h);
        l4 = f2(x + h, uv[0][i] + k3 * h, uv[1][i] + l3 * h);

        uv[0][i + 1] = uv[0][i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        uv[1][i + 1] = uv[1][i] + h / 6 * (l1 + 2 * l2 + 2 * l3 + l4);

        x += h;
    }

    return uv;
}

int main(int argc, char *argv[])
{
    int n, num;
    double a, b, yy0, u0, v0, alpha;

    printf("The differential equations:\n1: y' = y-yx\n2:\ny' = 3y/2x - 3x^2/2(y-1)^2\n3:\ny' = y/x + x*sqrt(1 - y^2/x^2)\n4:\n2e^(-3x) + e^x/4\n");
    printf("Choose equation: ");
    scanf("%d", &num);

    printf("Type left border (x0): ");
    scanf("%lf", &a);
    printf("Type right border: ");
    scanf("%lf", &b);
    printf("Type y0 = y(x0): ");
    scanf("%lf", &yy0);
    printf("Type the approximation function values array size: ");
    scanf("%d", &n);
    printf("Type alpha (Runge-Kutta second-order method parameter): ");
    scanf("%lf", &alpha);

    void *function;
    switch (num) {
    case 1:
        function = func1;
        break;
    case 2:
        function = func2;
        break;
    case 3:
        function = func3;
        break;
    default:
        function = func4;
    }

    if (!fork()) {
        int fd = open("out.txt", O_WRONLY | O_APPEND | O_CREAT);
        dup2(fd, STDOUT_FILENO);

        double *rk2 = RK2(a, b, yy0, n, alpha, function);
        if (rk2 == NULL) memerr();
        printf("RK2, Numerical solution:\n\n");
        PrintFunc(rk2, a, (b - a) / n, n + 1);
        printf("\n");
        free(rk2);
        return 0;
    }
    wait(NULL);

    if (!fork()) {
        int fd = open("out.txt", O_WRONLY | O_APPEND | O_CREAT);
        dup2(fd, STDOUT_FILENO);
        double *rk4 = RK4(a, b, yy0, n, function);
        if (rk4 == NULL) memerr();
        printf("\nRK4, Numerical solution:\n");
        PrintFunc(rk4, a, (b - a) / n, n + 1);
        printf("\n");
        free(rk4);
        return 0;
    }
    wait(NULL);

    printf("The systems of differential equations:\n1:\nu' = e^(-u^2-v^2) + 2x\nv' = 2u^2 + v\n2:\nu' = (u - v) / x\nv' = 2v / x + ux\n");
    printf("3:\nu' = 1/2u\nv' = v\n4:\nu' = (u - 3x)ln2 + 3\nv' = 1/x\n5:\nu' = 2e^(v - 3)\nv' = u - 2e^x + 1\n");
    printf("Choose system: ");
    scanf("%d", &num);
    printf("Type left border (x0): ");
    scanf("%lf", &a);
    printf("Type right border: ");
    scanf("%lf", &b);
    printf("Type u0 = u(x0): ");
    scanf("%lf", &u0);
    printf("Type v0 = v(x0): ");
    scanf("%lf", &v0);
    printf("Type the approximation functions values array size: ");
    scanf("%d", &n);
    printf("Type alpha (Runge-Kutta second-order method parameter): ");
    scanf("%lf", &alpha);

    void *function1, *function2;
    switch (num) {
    case 1:
        function1 = sysf11;
        function2 = sysf12;
        break;
    case 2:
        function1 = sysf21;
        function2 = sysf22;
        break;
    case 3:
        function1 = sysf31;
        function2 = sysf32;
        break;
    case 4:
        function1 = sysf41;
        function2 = sysf42;
        break;
    default:
        function1 = sysf51;
        function2 = sysf52;
    }

    if (!fork()) {
        int fd = open("out1.txt", O_WRONLY | O_APPEND | O_CREAT);
        dup2(fd, STDOUT_FILENO);
        double **rk2 = RKS2(a, b, u0, v0, n, alpha, function1, function2);
        if (rk2 == NULL) memerr();
        printf("RK2, Numerical solution:\nu(x):\n");
        PrintFunc(rk2[0], a, (b - a) / n, n + 1);
        printf("\nv(x):\n");
        PrintFunc(rk2[1], a, (b - a) / n, n + 1);
        printf("\n");
        free(rk2);
        return 0;
    }
    wait(NULL);

    if (!fork()) {
        int fd = open("out1.txt", O_WRONLY | O_APPEND | O_CREAT);
        dup2(fd, STDOUT_FILENO);
        double **rk4 = RKS4(a, b, u0, v0, n, function1, function2);
        if (rk4 == NULL) memerr();
        printf("RK4, Numerical solution:\nu(x):\n");
        PrintFunc(rk4[0], a, (b - a) / n, n + 1);
        printf("\nv(x):\n");
        PrintFunc(rk4[1], a, (b - a) / n, n + 1);
        printf("\n");
        free(rk4);
        return 0;
    }
    wait(NULL);

    return 0;
}
