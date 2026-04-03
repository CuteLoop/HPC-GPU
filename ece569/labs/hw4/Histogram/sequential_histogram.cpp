// bins as 4 letter sections of the alphabet, so 26/4 = 6.5, so we can have 7 bins
// bin 0: a-f, bin 1: g-l, bin 2: m-r, bin 3: s-x, bin 4: y-z
#include <iostream>


void sequential_histogram(char* data, int length, int* histogram) {
    // Initialize the histogram array to zero
    for (int i=0; i<length; i++) {
        int alphabet_position = data[i] - 'a'; // Get the position of the character in the alphabet (0 for 'a', 1 for 'b', ..., 25 for 'z')
        if (alphabet_position>=0 && alphabet_position<26) {
           histogram[alphabet_position/4]++; 
           //eg. go to bin 0 if 3/4 =0, go to bin 1 if 7/4 = 1, go to bin 2 if 15/4 = 3, etc.
        }
    }
}

