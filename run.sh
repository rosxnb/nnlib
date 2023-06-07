cmake -DCMAKE_BUILD_TYPE=Debug -Wdev -Wdeprecated -S $(pwd) -B "$(pwd)/build"

if [[ $? -eq 0 ]]
then
    echo "Executing generated Makefile"
    cd build
    make
    cd ..
fi

if [[ $? -eq 0 ]]
then
    echo "\nRunning the executable \n"
    ./build/exe
fi
