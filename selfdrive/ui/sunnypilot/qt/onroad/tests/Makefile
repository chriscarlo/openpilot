
CXX = g++
CXXFLAGS = -std=c++17 -I$(PROJECT_ROOT) -I$(PROJECT_ROOT)/third_party/googletest/include
LDFLAGS = -lgtest -lgtest_main -pthread

test_rti_arrow: test_rti_arrow_bearing.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f test_rti_arrow

.PHONY: clean
