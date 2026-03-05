void saxpy(int n, float a, float * restrict x, float * restrict y) {
    for (int i=0; i<n; i++){
        y[i] = a*x[i] + y[i];
    }
}

saxpy(1<<20, 2.0f, x, y);