test: test.cpp
	g++ -Wall -O2 -o test test.cpp

clean:
	rm -f test *.o
