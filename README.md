# Thị giác máy tính - BT2 - Edge Detection

## Thành viên
| MSSV | Họ và tên
|--|--
|1712060 | Trần Vinh Hưng
|1712786 | Nguyễn Văn Thiều

## Hướng dẫn compile bằng CMake

Step0 (need when run unit test): we need to clone `googletest` at [link](https://github.com/google/googletest/) to `lib/` 
to run unit test in debug mode.

Your directory tree should look like this:
```
.
|- Source
|- Test
|- lib
    |- googletest

```
 
Step1: Recompile source code.
```
# Create build directory
mkdir build && cd build

# Run CMake. <value> can be <Debug> or <Release>
cmake -DCMAKE_BUILD_TYPE=<value> ..

# Compile
make 
```

Step2: Run binary file:

```
# In build director
## Run Test
./Test/EdgeDetection_tst

## Run binary file
./Source/EdgeDetection_run
```

## Edit code and recompile
- `Source`: Contains all source code.
- `Test`: Contatins testcase

- Run `make EdgeDetection_tst` to recompile test case.
- Run `make EdgeDetection_run` to recompile main file.

## Contributing
- Link code convention: [Google Style Guide](https://google.github.io/styleguide/cppguide.html)

Note: Make sure you are using 2-space indent.

