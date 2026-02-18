// profile it. https://github.com/harrism/nsys_easy



| Command                           | Purpose                  |
| --------------------------------- | ------------------------ |
| `module load cuda11/11.0`         | Load CUDA tools          |
| `nvcc -O2 file.cu -o exe`         | Compile CUDA program     |
| `./exe`                           | Run program              |
| `nvprof ./exe`                    | Basic GPU profiling      |
| `nvprof --log-file out.txt ./exe` | Save profile to file     |
| `nvprof -o profile.nvvp ./exe`    | Create GUI timeline file |
| `cuda-memcheck ./exe`             | Check memory errors      |
| `salloc --gres=gpu:1`             | Request GPU node         |
